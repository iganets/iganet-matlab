# IgANets-Template

[![GitlabSync](https://github.com/IgANets/iganet-template/actions/workflows/gitlab-sync.yml/badge.svg)](https://github.com/IgANets/iganet-template/actions/workflows/gitlab-sync.yml)
[![CMake on multiple platforms](https://github.com/IgANets/iganet-template/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/IgANets/iganet-template/actions/workflows/cmake-multi-platform.yml)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://iganets.github.io/iganet/)

[![GitHub Releases](https://img.shields.io/github/release/iganets/iganet-template.svg)](https://github.com/iganets/iganet-template/releases)
[![GitHub Downloads](https://img.shields.io/github/downloads/iganets/iganet-template/total)](https://github.com/iganets/iganet-template/releases)
[![GitHub Issues](https://img.shields.io/github/issues/iganets/iganet-template.svg)](https://github.com/iganets/iganet-template/issues)

This repository serves as a template for standalone applications for [IgANets](https://github.com/iganets/iganet), a novel approach to combine the concept of deep operator learning with the mathematical framework of isogeometric analysis.

## Setup instructions

Don't commit to this repository as it serves as a generic template repository. Instead, clone it for your application, say, https://github.com/IgANets/iganet-myapp and adapt it as follows:

1. Change `iganet-template` to `iganet-myapp` in the `README.md` file

2. Change `TEMPLATE` to `MyApp` in the `CMakeLists.txt` file

3. Implement your application(s) in the `src` directory.

_Optional:_

4. By default, the template repository contains the following [GitHub actions](https://github.com/features/actions)

   - `.github/workflows/gitlab-sync.yml` synchronizes commits between GitHub and GitLab. If you want this synchronization to work you need to create a repository at https://gitlab.com and set the following [repository secrets](https://docs.github.com/en/actions/how-tos/writing-workflows/choosing-what-your-workflow-does/using-secrets-in-github-actions)
     | repository secret | content |
     |---|---|
     | `GITLAB_REPO_USERNAME` | username that enables commits to the repository |
     | `GITLAB_REPO_TOKEN` | token that enables commits to the repository |
     | `GITLAB_REPO_URL`   | https://gitlab.com/organization/repository |
     
     Otherwise, delete this file.
   - `.github/workflows/cmake-multi-platform.yml` basic skeleton for multi-platform CI/CD

## Usage instructions

The `CMakeLists.txt` file of this template repository is set up in a way that it downloads the latest master version of [IgANets](https://github.com/iganets/iganet) as dependency and imports the target `iganet::core`. To configure and build a standalone executable for each source file in the `src` directory follow the instructions below:

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
