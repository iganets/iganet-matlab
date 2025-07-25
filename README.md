# iganets-Matlab

[![GitlabSync](https://github.com/iganets/iganet-matlab/actions/workflows/gitlab-sync.yml/badge.svg)](https://github.com/iganets/iganet-matlab/actions/workflows/gitlab-sync.yml)
[![CMake on multiple platforms](https://github.com/iganets/iganet-matlab/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/iganets/iganet-matlab/actions/workflows/cmake-multi-platform.yml)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://iganets.github.io/iganet/)

[![GitHub Releases](https://img.shields.io/github/release/iganets/iganet-matlab.svg)](https://github.com/iganets/iganet-matlab/releases)
[![GitHub Downloads](https://img.shields.io/github/downloads/iganets/iganet-matlab/total)](https://github.com/iganets/iganet-matlab/releases)
[![GitHub Issues](https://img.shields.io/github/issues/iganets/iganet-matlab.svg)](https://github.com/iganets/iganet-matlab/issues)

This repository contains the MATLAB bindings for [IGAnets](https://github.com/iganets/iganet), a novel approach to combine the concept of deep operator learning with the mathematical framework of isogeometric analysis.

## Usage instructions

The `CMakeLists.txt` file of this template repository is set up in a way that it downloads the latest master version of [iganets](https://github.com/iganets/iganet) as dependency and imports the target `iganet::core`. To configure and build the MATLAB MEX function source file in the `src` directory follow the instructions below:

1. Create a `build` directory
   ```shell
   mkdir build
   ```

2. Configure `cmake`
   ```shell
   cmake <path-to-iganet-template-directory>
   ```

3. Compile the code
   ```shell
   make
   ```
