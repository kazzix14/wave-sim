# wave-simulator

This repository is remnant of my struggle to reproduce aerophone simulation software presented by [this paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/Aerophones.pdf).
It can be run, but it does not work as a aerophone simulator. As my interest goes to [this](https://github.com/kazzix14/raytrace-reverb) and this repository is left WIP.

![wave-sim3](https://user-images.githubusercontent.com/29710855/104088201-6193ed80-52a8-11eb-8ff3-e09289732460.gif)

# How to use

Run

    cargo run --release

or

    cargo run --release --features gui

When gui feature enable, it make window so that you can see the waves and change settings.  
Middle click to toggle blow state. Right click to set blow pressure (Y coordinate of the cursor is gonna be pressure value. top is origin.). Left click to draw walls.

# LISCENSE

This project is licensed under the Mozilla Public License, v. 2.0 - see the [LICENSE](LICENSE) file for details
