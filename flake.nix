{
  description = "Mindclade reproducible developer shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        commonPackages = with pkgs; [
            bazelisk
            buf
            cargo
            cosign
            docker-client
            docker-buildx
            go
            just
            kubectl
            kustomize
            python312
            rustc
            uv
          ];
      in {
        devShells = {
          default = pkgs.mkShell {
            packages = commonPackages;
            shellHook = ''
              export UV_PYTHON="${pkgs.python312}/bin/python"
              export PYTHONNOUSERSITE=1
            '';
          };
        } // pkgs.lib.optionalAttrs (system == "x86_64-linux") {
          cuda = pkgs.mkShell {
            packages = commonPackages ++ (with pkgs.cudaPackages_13_0; [
              cuda_cudart
              cuda_nvcc
            ]);
            CUDA_HOME = pkgs.cudaPackages_13_0.cuda_nvcc;
            MINDCLADE_CUDA_PROFILE = "13.0";
            shellHook = ''
              export UV_PYTHON="${pkgs.python312}/bin/python"
              export PYTHONNOUSERSITE=1
            '';
          };
        };
      });
}
