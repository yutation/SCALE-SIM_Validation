import math
import random
from typing import List, Tuple, Optional, Set

Shape3D = Tuple[int, int, int]

def _log_uniform_int(lo: int, hi: int, rng: random.Random) -> int:
    """Sample integer roughly log-uniform in [lo, hi]."""
    assert lo >= 1 and hi >= lo
    a = math.log(lo)
    b = math.log(hi)
    x = math.exp(rng.uniform(a, b))
    v = int(round(x))
    return max(lo, min(hi, v))

def _random_divisor(n: int, rng: random.Random, min_val: int = 1) -> int:
    """Pick a random divisor of n (not necessarily uniform over divisors, but good enough).
    
    Args:
        n: The number to find a divisor of.
        rng: Random number generator.
        min_val: Minimum value for the divisor (default 1). Use min_val=2 to avoid trivial divisors.
    """
    sqrt_n = int(math.isqrt(n))
    # Try a few random trials by sampling a factor candidate
    for _ in range(50):
        lo = max(min_val, 1)
        hi = max(lo, sqrt_n)
        d = rng.randint(lo, hi)
        if n % d == 0:
            other = n // d
            # Return the divisor or its complement, but respect min_val for both
            if d >= min_val and other >= min_val:
                return d if rng.random() < 0.5 else other
            elif d >= min_val:
                return d
            elif other >= min_val:
                return other
    # Fallback: simple scan (rarely used)
    divs = [d for d in range(min_val, sqrt_n + 1) if n % d == 0]
    if not divs:
        return n  # n itself is the only divisor >= min_val
    d = rng.choice(divs)
    other = n // d
    if other >= min_val:
        return d if rng.random() < 0.5 else other
    return d

def _factor_to_3d(numel: int, rng: random.Random, dim: int) -> Shape3D:
    """
    Turn numel into a (d0,d1,d2) with product <= numel (ideally == numel).
    dim in {1,2,3}: force dimensionality; missing dims filled with 1.
    """
    assert dim in (1, 2, 3)
    if dim == 1:
        return (numel, 1, 1)

    if dim == 2:
        d0 = _random_divisor(numel, rng)
        d1 = numel // d0
        # Randomly swap
        if rng.random() < 0.5:
            d0, d1 = d1, d0
        return (d0, d1, 1)

    # dim == 3: generate 3 independent dimensions, each >= 2
    # This guarantees a true 3D shape instead of trying to factor numel
    # Compute max per-dimension size: cube root of numel
    max_per_dim = max(2, int(numel ** (1/3)))
    d0 = _log_uniform_int(2, max_per_dim, rng)
    d1 = _log_uniform_int(2, max_per_dim, rng)
    # For d2, constrain to keep total numel reasonable
    max_d2 = max(2, numel // (d0 * d1))
    d2 = _log_uniform_int(2, max_d2, rng) if max_d2 >= 2 else 2
    dims = [d0, d1, d2]
    rng.shuffle(dims)
    return (dims[0], dims[1], dims[2])

def _clamp_shape_to_max_numel(shape: Shape3D, max_numel: int) -> Shape3D:
    """If product too big, shrink the largest dim to fit (keeps shape 'nearby')."""
    d0, d1, d2 = shape
    prod = d0 * d1 * d2
    if prod <= max_numel:
        return shape
    dims = [d0, d1, d2]
    i = max(range(3), key=lambda k: dims[k])
    other = (prod // dims[i])
    # set dims[i] <= floor(max_numel / other)
    new_i = max(1, max_numel // other)
    dims[i] = new_i
    return (dims[0], dims[1], dims[2])

def generate_shapes_for_latency(
    n_shapes: int,
    max_numel: int = 16 * 1024 * 1024,
    dim_probs: Tuple[float, float, float] = (0.25, 0.55, 0.20),  # (1D,2D,3D)
    boundary_frac: float = 0.30,
    boundary_mods: Tuple[int, ...] = (8, 16, 32, 64, 128, 256),
    perturb_range: int = 32,
    seed: int = 0,
    ensure_unique: bool = True,
) -> List[Shape3D]:
    """
    Generate a good measurement dataset of shapes for elementwise op latency modeling.

    - <=3D shapes represented as (d0,d1,d2); unused dims are 1.
    - numel <= max_numel.
    - Mixture of log-uniform random shapes + boundary-focused perturbations
      (to capture alignment / tiling / launch edge effects).

    Returns: list of shape tuples (d0,d1,d2).
    """
    rng = random.Random(seed)

    # Normalize dim_probs
    p1, p2, p3 = dim_probs
    s = p1 + p2 + p3
    p1, p2, p3 = p1 / s, p2 / s, p3 / s

    def sample_dim() -> int:
        r = rng.random()
        if r < p1:
            return 1
        elif r < p1 + p2:
            return 2
        else:
            return 3

    n_boundary = int(round(n_shapes * boundary_frac))
    n_random = n_shapes - n_boundary

    shapes: List[Shape3D] = []
    seen: Set[Shape3D] = set()

    def add_shape(sh: Shape3D):
        sh = _clamp_shape_to_max_numel(sh, max_numel)
        if sh[0] < 1 or sh[1] < 1 or sh[2] < 1:
            return
        if sh[0] * sh[1] * sh[2] > max_numel:
            return
        # Reject invalid "3D" shapes that have a 1 in a non-trailing position
        # Valid patterns: (x,1,1) for 1D, (x,y,1) for 2D, (x,y,z) with all >1 for 3D
        d0, d1, d2 = sh
        if d2 > 1 and (d0 == 1 or d1 == 1):
            return  # Invalid 3D shape
        if d1 > 1 and d0 == 1 and d2 == 1:
            # Swap to make it valid 2D: (1,y,1) -> (y,1,1) is not a fix, just reject
            # Actually (1,y,1) is fine as 2D if we consider d1 as the meaningful dim
            pass  # This is acceptable as 2D
        if ensure_unique:
            if sh in seen:
                return
            seen.add(sh)
        shapes.append(sh)

    # 1) Random shapes: log-uniform in numel, then factor to 1D/2D/3D
    # Use dimension-appropriate minimum numels to ensure proper factorization
    min_numel_for_dim = {1: 1, 2: 2, 3: 8}  # 3D needs at least 2*2*2=8
    tries = 0
    while len(shapes) < n_random and tries < n_random * 50:
        tries += 1
        dim = sample_dim()
        min_numel = min_numel_for_dim[dim]
        if min_numel > max_numel:
            min_numel = 1  # fallback for very small max_numel
        numel = _log_uniform_int(min_numel, max_numel, rng)
        sh = _factor_to_3d(numel, rng, dim)
        add_shape(sh)

    # 2) Boundary-focused: start from an existing or fresh random shape, then perturb dims
    # Strategy: pick a modulus M, try to push some dimension near multiples of M (± perturb_range)
    tries = 0
    while len(shapes) < n_shapes and tries < n_shapes * 200:
        tries += 1
        # Base shape: prefer existing ones (more realistic local neighborhoods)
        if shapes and rng.random() < 0.7:
            base = rng.choice(shapes)
        else:
            dim = sample_dim()
            min_numel = min_numel_for_dim[dim]
            if min_numel > max_numel:
                min_numel = 1
            numel = _log_uniform_int(min_numel, max_numel, rng)
            base = _factor_to_3d(numel, rng, dim)

        d = list(base)
        # Choose which dim(s) to perturb (ignore dims that are 1 with some probability)
        idxs = [0, 1, 2]
        rng.shuffle(idxs)
        k = 1 if rng.random() < 0.8 else 2  # perturb 1 dim most of the time
        idxs = idxs[:k]

        M = rng.choice(boundary_mods)

        # Determine minimum value for each dimension based on base shape's dimensionality
        # If base is 3D (no 1s), keep all dims >= 2; if 2D (one 1), keep non-1 dims >= 2
        base_is_3d = all(x > 1 for x in base)
        base_is_2d = sum(1 for x in base if x == 1) == 1
        
        for i in idxs:
            # if dim is 1, sometimes skip (keeps 1D/2D distribution)
            if d[i] == 1 and rng.random() < 0.6:
                continue

            # Move dimension near a multiple of M with a random offset
            # target = round(d[i]/M)*M + offset
            offset = rng.randint(-perturb_range, perturb_range)
            target_mult = int(round(d[i] / M)) * M
            # For 3D shapes, keep dims >= 2; for others, allow 1
            min_val = 2 if (base_is_3d or (base_is_2d and base[i] > 1)) else 1
            new_di = max(min_val, target_mult + offset)
            d[i] = new_di

        sh = (d[0], d[1], d[2])
        add_shape(sh)

    # If still short (due to uniqueness constraints), top up with pure random
    tries = 0
    while len(shapes) < n_shapes and tries < n_shapes * 200:
        tries += 1
        numel = _log_uniform_int(1, max_numel, rng)
        sh = _factor_to_3d(numel, rng, sample_dim())
        add_shape(sh)

    return shapes[:n_shapes]


def generate_shapes_simple_2d(
    n_shapes: int,
    max_numel: int = 16 * 1024 * 1024,
    dim_probs: Tuple[float, float] = (0.3, 0.7),  # (1D, 2D)
    seed: int = 0,
    ensure_unique: bool = True,
) -> List[Shape3D]:
    """
    Simplified 2D shape generation with uniform distribution.
    
    - Generates only 1D and 2D shapes
    - 1D: uniform distribution of sizes
    - 2D: uniform distribution of total size, then random factorization
    - dim_probs controls the proportion of 1D vs 2D shapes
    
    Args:
        n_shapes: Total number of shapes to generate
        max_numel: Maximum number of elements per shape
        dim_probs: (prob_1D, prob_2D) - will be normalized
        seed: Random seed
        ensure_unique: Whether to ensure all shapes are unique
        
    Returns: list of shape tuples (d0,d1,d2) where d2 is always 1
    """
    rng = random.Random(seed)
    
    # Normalize dim_probs
    p1, p2 = dim_probs
    s = p1 + p2
    p1, p2 = p1 / s, p2 / s
    
    # Calculate how many shapes of each dimension
    n_1d = int(round(n_shapes * p1))
    n_2d = n_shapes - n_1d
    
    shapes: List[Shape3D] = []
    seen: Set[Shape3D] = set()
    
    def add_shape(sh: Shape3D) -> bool:
        """Try to add a shape, return True if successful."""
        if sh[0] < 1 or sh[1] < 1 or sh[2] < 1:
            return False
        if sh[0] * sh[1] * sh[2] > max_numel:
            return False
        if ensure_unique:
            if sh in seen:
                return False
            seen.add(sh)
        shapes.append(sh)
        return True
    
    # Generate 1D shapes with uniform distribution
    tries = 0
    while len([s for s in shapes if s[1] == 1 and s[2] == 1]) < n_1d and tries < n_1d * 100:
        tries += 1
        size = rng.randint(1, max_numel)
        sh = (size, 1, 1)
        add_shape(sh)
    
    # Generate 2D shapes with uniform total size, then random factorization
    tries = 0
    while len([s for s in shapes if s[1] > 1]) < n_2d and tries < n_2d * 100:
        tries += 1
        # Uniform distribution of total size
        total_size = rng.randint(2, max_numel)
        
        # Random factorization: pick a random divisor
        d0 = _random_divisor(total_size, rng, min_val=1)
        d1 = total_size // d0
        
        # Randomly swap dimensions
        if rng.random() < 0.5:
            d0, d1 = d1, d0
        
        sh = (d0, d1, 1)
        add_shape(sh)
    
    # If we're short due to uniqueness constraints, fill with random shapes
    tries = 0
    while len(shapes) < n_shapes and tries < n_shapes * 100:
        tries += 1
        if rng.random() < p1:
            # 1D shape
            size = rng.randint(1, max_numel)
            sh = (size, 1, 1)
        else:
            # 2D shape
            total_size = rng.randint(2, max_numel)
            d0 = _random_divisor(total_size, rng, min_val=1)
            d1 = total_size // d0
            if rng.random() < 0.5:
                d0, d1 = d1, d0
            sh = (d0, d1, 1)
        add_shape(sh)
    
    return shapes[:n_shapes]


if __name__ == "__main__":
    # Test the new simple 2D generation
    shapes = generate_shapes_simple_2d(
        n_shapes=1000,
        max_numel=1 * 1024 * 1024,
        dim_probs=(0.3, 0.7),  # 30% 1D, 70% 2D
        seed=42,
    )
    print("num shapes:", len(shapes))
    print("first 10:", shapes[:10])
    
    # Count 1D vs 2D
    n_1d = sum(1 for s in shapes if s[1] == 1 and s[2] == 1)
    n_2d = sum(1 for s in shapes if s[1] > 1)
    print(f"1D shapes: {n_1d} ({n_1d/len(shapes)*100:.1f}%)")
    print(f"2D shapes: {n_2d} ({n_2d/len(shapes)*100:.1f}%)")
    
    # Sanity: max numel
    print("max numel in set:", max(a*b*c for a,b,c in shapes))
