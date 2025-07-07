# New Project Overview

This project implements various reinforcement learning algorithms, including Proximal Policy Optimization (PPO) and Direct Preference Optimization (DPO). It provides a structured approach to training and evaluating machine learning models using both synthetic and real-world datasets.

## Project Structure

- **models/**: Contains scripts for loading machine learning models.
  - `load_model.py`: Responsible for loading models from specified paths.

- **trainers/**: Implements training algorithms.
  - `ppo_trainer.py`: Contains the PPO training algorithm.
  - `dpo_trainer.py`: Contains the DPO training algorithm.

- **reward_models/**: Defines the reward model used in training.
  - `reward_model.py`: Implements the reward calculation logic.

- **data/**: Holds datasets for training and testing.
  - `synthetic/`: Directory for synthetic datasets.
  - `real/`: Directory for real-world datasets.

- **scripts/**: Contains scripts for training and evaluation.
  - `train_ppo.py`: Script to initiate PPO training.
  - `train_dpo.py`: Script to initiate DPO training.
  - `evaluate.py`: Script to evaluate trained models.

- **ui/**: Contains the Streamlit application for user interaction.
  - `streamlit_app.py`: Implements the user interface for visualizing results.

- **configs/**: Configuration files for training settings.
  - `ppo.yaml`: YAML file containing hyperparameters for PPO training.

- **requirements.txt**: Lists the required Python packages for the project.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd new-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare your datasets:
   - Place synthetic datasets in the `data/synthetic/` directory.
   - Place real-world datasets in the `data/real/` directory.

## Usage Guidelines

- To train the PPO model, run:
  ```
  python scripts/train_ppo.py
  ```

- To train the DPO model, run:
  ```
  python scripts/train_dpo.py
  ```

- To evaluate the trained models, run:
  ```
  python scripts/evaluate.py
  ```

- To launch the Streamlit application, run:
  ```
  streamlit run ui/streamlit_app.py
  ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.