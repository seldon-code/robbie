project('seldon', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=c++20', ])

add_global_arguments('-Wno-unused-local-typedefs', language : 'cpp')

incdir = include_directories('include')

tests = [
  ['Test1', 'test/test1.cpp'],
]

Catch2 = dependency('Catch2', method : 'cmake', modules : ['Catch2::Catch2WithMain', 'Catch2::Catch2'])

foreach t : tests
  exe = executable(t.get(0), t.get(1),
    dependencies : [dependency('fmt'), Catch2],
    include_directories : incdir
  )
  test(t.get(0), exe)
endforeach

exe = executable('robbie', 'main.cpp',
  install : true,
  dependencies : [dependency('fmt')],
  include_directories : incdir
  )
