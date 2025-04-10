from autotrain.params import Seq2SeqParams
from autotrain.project import AutoTrainProject
params = Seq2SeqParams(
    data_path="data/",
    train_split = "train",
    model="google/mt5-small",
    text_column="javanese",
    target_column="indonesian",
    lr=2e-5,
    batch_size=2,
    epochs=10,
    max_seq_length=256,
    max_target_length=256,
    # Add gradient clipping
    max_grad_norm=1.0,
    project_name="mt5-small-translation-seq2seq-8",
    username="ttmenezes",
    push_to_hub=True,
    token=""
)

project = AutoTrainProject(params=params, backend="local", process=True)
project.create()