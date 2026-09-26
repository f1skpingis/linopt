# Simplex Algorithm Implementation

For the current implementation, start with the
[simplex state and catch-up guide](../books/simplex-state-and-guide.md) and
[Forrest-Tomlin TODO](../books/forresttomlintodo.md), audited on 2026-09-13.
The assignment description below is historical and refers to older filenames
and inverse-update tasks.

## Introduction

Welcome to the Simplex Algorithm programming assignment! The goal of this assignment is to implement a robust version of the revised simplex method for solving linear programming problems on standard form:

Minimize `c^T * x`,
subject to `A * x = b`, `x >= 0.`

You will be filling in the core components of the algorithm, including the logic for updating the basis inverse, implementing a pivoting strategy, and orchestrating the main solver loop.
All necessary information and math can be found in chapter 13, *Linear Programming: The Simplex Algorithm*, of the book.

## Project Structure

The project is organized into two main packages:

- `src/simplex`: This is your workspace. You will find skeleton files with `TODO` markers indicating where you need to write your implementation.
- `src/simplex`: This directory contains a working implementation of the algorithm. It's there for you to use as the starting implementation. I am happy to take suggestions of improvements if you disapprove of my implementation.

The tests are designed to work with your code in `src/simplex` as you complete the `TODO`s.

## Setup

To get started, you need [uv](https://docs.astral.sh/uv/getting-started/) installed.

1. **Create a virtual environment:**
    From the `linopt/simplex` directory, run:

    ```bash
    uv sync
    ```

    This will create a `.venv` directory with all the dependencies installed.

2. **Open project and select the virtual environment in VSCode**
    Open the folder `linopt/simplex` in VSCode.
    Select interpreter and chose python in the `.venv`.

## Your Task

Your task is to complete the implementation in the `src/simplex` directory. Look for the `TODO` comments in the following files:

### `math.py`

- **`update_inverse`**: Implement a numerically efficient method to update the inverse of the basis matrix (`Binv`) after a pivot operation. This is a core component of the revised simplex method.

### `pivoting_strategy.py`

- **`MyPivotStrategy`**: Implement a pivoting strategy that guarantees finite termination. You will need to implement:
  - `pick_entering_index`: Choose a variable to enter the basis from the non-basic variables with negative reduced costs.
  - `pick_exiting_index`: Choose a variable to leave the basis by applying the ratio test.

### `solver.py`

- **`find_basic_feasible_solution`**: Implement the logic for Phase I of the simplex method. This involves setting up and solving an auxiliary linear program to find an initial basic feasible solution for the original problem.
- **`solve`**: Complete the main solver loop. This involves:
    1. Calculating the reduced costs for non-basic variables.
    2. Using your pivoting strategy to select entering and exiting variables.
    3. Updating the basis inverse using your `update_inverse` function.
    4. Updating the basic solution `x_basis`.

## Running the Tests

A test suite is provided in the `tests/` directory to help you verify your implementation. The tests cover individual components like the pivoting strategy as well as the full solver on various problems.

To run the tests, make sure your virtual environment is activated and run the following command from the `linopt/simplex` directory:

```bash
pytest .
```

or use the test GUI in VSCode.

The tests are designed to pass one by one as you correctly implement the `TODO`s.

## Resources

The implementation details for the revised simplex method, the two-phase simplex method (Phase I), and anti-cycling rules can, as mentioned, be found in chapter 13 of *Numerical Optimization*.
