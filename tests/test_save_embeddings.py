import pickle
import sys
import types
from pathlib import Path

# Stub external dependencies required by reve_model
torch_module = types.ModuleType("torch")
sys.modules.setdefault("torch", torch_module)

mne_module = types.ModuleType("mne")
mne_module.io = types.SimpleNamespace(
    BaseRaw=object, read_raw_brainvision=lambda *a, **k: None
)
sys.modules.setdefault("mne", mne_module)

mne_bids = types.ModuleType("mne_bids")

class DummyBIDSPath:
    def __init__(*args, **kwargs):
        pass

mne_bids.BIDSPath = DummyBIDSPath
sys.modules["mne_bids"] = mne_bids

goofi = types.ModuleType("goofi")
goofi_data = types.ModuleType("goofi.data")
goofi_data.to_data = lambda *args, **kwargs: None
goofi_nodes = types.ModuleType("goofi.nodes")
goofi_nodes_analysis = types.ModuleType("goofi.nodes.analysis")
goofi_reveeeg = types.ModuleType("goofi.nodes.analysis.reveeeg")
goofi_reveeeg.ReveEEG = object
sys.modules["goofi"] = goofi
sys.modules["goofi.data"] = goofi_data
sys.modules["goofi.nodes"] = goofi_nodes
sys.modules["goofi.nodes.analysis"] = goofi_nodes_analysis
sys.modules["goofi.nodes.analysis.reveeeg"] = goofi_reveeeg

tqdm_module = types.ModuleType("tqdm")
tqdm_module.tqdm = lambda x, **kwargs: x
sys.modules["tqdm"] = tqdm_module

# Ensure the project root is on the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dl_pipelines.reve_model import save_embeddings


def test_save_embeddings_writes_to_output(tmp_path):
    """Ensure save_embeddings writes file to the specified directory."""
    output_dir = tmp_path / "sub-0001"
    embeddings = {0: [1, 2, 3]}
    save_embeddings(
        embeddings,
        output_dir,
        "0001",
        segment_duration=60,
        z_score=False,
        z_score_axis=1,
        notch=False,
    )
    expected_file = output_dir / "sub-0001_task-RESTING_run-01_embeddingsseg60.pkl"
    assert expected_file.exists(), "Embedding file was not created in the expected directory"
    with open(expected_file, "rb") as f:
        loaded = pickle.load(f)
    assert loaded == embeddings

