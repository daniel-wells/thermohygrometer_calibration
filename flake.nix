{
  description = "Nix development shell for thermo-hygrometer calibration";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python312;
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.uv
            pkgs.quarto
            pkgs.gcc
            pkgs.pkg-config
          ]
          ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
            pkgs.libiconv
          ];

          env = {
            UV_PYTHON = "${python}/bin/python";
          };

          shellHook = ''
            # Keep uv virtualenv outside Dropbox-backed repo directories.
            export UV_PROJECT_ENVIRONMENT="$HOME/.cache/uv/venvs/$(basename "$PWD")"
            # Force Quarto code execution to use the uv-managed interpreter.
            export QUARTO_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"

            echo "Thermo-hygrometer dev shell ready."
            echo "uv environment: $UV_PROJECT_ENVIRONMENT"
            echo "quarto python: $QUARTO_PYTHON"
            echo "1) uv sync"
            echo "2) uv run python -m thermohygrometer_calibration.simulate --layout data/layout.csv --output-dir data/simulated"
          '';
        };
      });
}
