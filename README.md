# PageRank
PageRank implementace v jazyce **C++ 20**. Tato implementace 
dosahuje warp-rychlostí při paralelním načítání a výpočtu
s využitím následujících technologií:
- **OpenMP**
- **Intel AVX2** SIMD instrukce

## Použití

### Sestavení
```bash
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build .
```

### Spuštění
Program byl testován na síti `web-BerkStan.txt.gz`
```bash
$ ./PageRank
```


## Licence
BSD 2-Clause License

Copyright (c) 2025, Daniel Slavík

[wortelus.eu](https://wortelus.eu)
