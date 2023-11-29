APP_TITLE = "RoP Detection"

# All dirs path relative to src folder

MODEL_H5_FOLDER = "../models/"
MODEL_H5_FILES_AND_LABELS = {
    "Stage Classification (All Images).h5": ['stage 1', 'stage 2', 'stage 3'],
    "Stage Classification (Temporal Images).h5": ['stage 1', 'stage 2', 'stage 3'],
    "Stage Classification (Manual Annotated).h5": ['stage 2', 'stage 3'],
    "Stage Classification (Manual + Temporal).h5": ['stage 2', 'stage 3'],
    "Decision Classification.h5": ["needs urgent treatment", "needs follow up", "discharged from rop"]
}
