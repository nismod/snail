# Tests

*Important*: The following commands must be run from the top-level directory.

Build the tests executable with

```shell
cmake -Bbuild .
cmake --build build/
```

Run the tests with

```shell
# In directory src/cpp/tests
./build/tests
```

The test executable accepts a range of option (thanks to [Cath2](https://github.com/catchorg/Catch2)). You can list them with

```shell
./build/tests --help
```

