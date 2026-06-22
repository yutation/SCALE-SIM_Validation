module @jit_simple_gemm_mul attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<128x256xbf16>, %arg1: tensor<256x512xbf16>, %arg2: tensor<128x512xbf16>) -> (tensor<128x512xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x256xbf16>, tensor<256x512xbf16>) -> tensor<128x512xbf16>
    %1 = stablehlo.multiply %0, %arg2 : tensor<128x512xbf16>
    return %1 : tensor<128x512xbf16>
  }
}
