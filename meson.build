project('seldon', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=c++20', 'optimization=3'])

add_global_arguments('-Wno-unused-local-typedefs', language : 'cpp')

incdir = include_directories('include')

exe = executable('robbie', 'main.cpp',
  install : true,
  dependencies : [dependency('fmt')],
  include_directories : incdir
  )

tests = [
  ['Test_ActivationFunctions', 'test/test_activation_functions.cpp'],
  ['Test_LossFunctions', 'test/test_loss_functions.cpp'],
  ['Test_FullyConnectedLayer', 'test/test_fc_layer.cpp'],
  ['Test_XOR', 'test/test_xor.cpp'],
  ['Test_BackwardPropagations', 'test/test_backward_propagations.cpp'],
]

Catch2 = dependency('Catch2', method : 'cmake', modules : ['Catch2::Catch2WithMain', 'Catch2::Catch2'])

foreach t : tests
  exe = executable(t.get(0), [t.get(1)],
    dependencies : [dependency('fmt'), Catch2],
    include_directories : [incdir, 'test']
  )
  test(t.get(0), exe)
endforeach


examples = [
  ['mnist', ['examples/mnist/main.cpp', 'examples/mnist/mnist_loader.cpp']]
]

foreach e : examples
  exe = executable(e.get(0), e.get(1), 
    dependencies : [dependency('fmt')], 
    include_directories :  ['examples' / e.get(0), incdir]
    )
endforeach
