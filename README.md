# Toward Optimal Tabletop Rearrangement with Multiple Manipulation Primitives

*[Baichuan Huang](https://baichuan05.github.io/), Xujia Zhang, [Jingjin Yu](https://arc-l.github.io/)*

### 3 minutues intro video:
https://github.com/arc-l/remp/assets/20850928/854b0654-f8e0-4ae3-837d-8b16d8b6522f


## Installation
* Docker (amd64)<br>
    ```
    git clone https://github.com/arc-l/remp.git
    cd remp
    docker pull baichuan55555/remp:latest
    bash run_container.sh
    ```

* Manually install (Ubuntu)<br>
    Refer to `install.sh` for a step-by-step installation

## Quick Start
* Case by case run:
  * `python environment.py --method mcts --case real_case_6_1 --gui`
* Benchmark:
  * `bash benchmark.sh`

## Test cases
All cases are stored in `sim_tests`.

Example:
Run `python environment.py --method mcts --case case_4_3 --gui`
A PyBullet GUI will pop up, the algorithm will plan the action, and the robot in the simulation will operate.