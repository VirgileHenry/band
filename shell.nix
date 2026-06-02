{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell rec {
  nativeBuildInputs = [
    pkgs.pkg-config

    pkgs.cargo
    pkgs.cargo-flamegraph
    pkgs.rustc
    pkgs.rustfmt
    pkgs.lld_20
    pkgs.mold

    pkgs.python313
    pkgs.python313Packages.numpy
    pkgs.python313Packages.librosa
    pkgs.python313Packages.mido
    pkgs.python313Packages.matplotlib
    pkgs.python313Packages.sounddevice
  ];
  buildInputs = [
    pkgs.udev
    pkgs.alsa-lib-with-plugins
    pkgs.vulkan-loader
    pkgs.xorg.libX11
    pkgs.xorg.libXcursor
    pkgs.xorg.libXi
    pkgs.xorg.libXrandr
    pkgs.libxkbcommon
    pkgs.wayland
  ];
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;
  RUST_BACKTRACE=1;
  TMPDIR="/tmp";
}
