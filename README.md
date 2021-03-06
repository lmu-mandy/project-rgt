# STEM Based TED Talk Generation
## Dataset
The dataset that was used for this project can be found on Kaggle: https://www.kaggle.com/miguelcorraljr/ted-ultimate-dataset. This dataset comes in several different languages, however we only used the English version which is titled `ted_talks_en.csv`.

## How to run the GPT-2 model
1. Clone this repository by running the following command: `git clone https://github.com/lmu-mandy/project-rgt`.
2. Open the `gpt_2.py` script and open the link which is located at the top of the file. This will open the Google Colab notebook.
3. Once you are in Google Colab, make a copy of the notebook, unless you do not intend to change anything in the code.
4. On the copy of the notebook, run the first cell which will install the `simpletransformers` package. You only need to run this cell once and should comment the line out after it has been run.
5. Navigate to the second cell in the notebook and add any prompt that you wish to generate STEM related text for.
6. Navigate to the `Runtime` button at the top of the notebook and click on `Change runtime type`. Under the `Hardware accelerator`, select `GPU`.
7. Navigate to the `Runtime` button once again and click `Restart and run all`. This will run all the cells in the notebook. After the model has been fine-tuned, the generated text will be printed at the bottom of the notebook for whatever prompts you entered.
8. If you want to change the prompts after the model has been trained, simply edit the prompts in the second cell, re-run that call, and then re-run the last cell to generate new text.
