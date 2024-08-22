from setuptools import setup, Extension
from torch.utils import cpp_extension
from glob import glob
from pathlib import Path
from vllm import (
    cache_ops,
    attention_ops,
    pos_encoding_ops,
)
from xformers import _C_flashattention
import sys
sys.argv = sys.argv+['develop']
CurPath = Path(__file__).parent.absolute()

setup(name='paged_attn',
      version='0.1.1',
      ext_modules=[cpp_extension.CppExtension(
          'paged_attn',
          sources=glob('custom_op_library/*.cc') +
           glob('custom_op_library/torch_extension_ort/*.cc'),
          extra_link_args=[
              cache_ops.__file__,
              attention_ops.__file__,
              pos_encoding_ops.__file__,
              _C_flashattention.__file__,
              #"/home/jicwen/work/xformers/build/lib.linux-x86_64-cpython-38/xformers/_C.so",
              #str(Path(_C_flashattention.__file__).parent/"_C.so"),
          ],
          include_dirs=[
              str(CurPath)+'/custom_op_library/onnxruntime-linux-x64-gpu-1.16.1/include'],
          #extra_compile_args=['-O0', '-g'],
      ),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
