# Tests

_Important_: The following commands must be run from the top-level directory in this repository.

Build the tests executable with

```shell
cmake -Bbuild ./extension
cmake --build build/
```

Run the tests with

```shell
./build/run_tests
```

The test executable accepts a range of options (thanks to [Catch2](https://github.com/catchorg/Catch2)). You can list them with

```shell
./build/run_tests --help
```
