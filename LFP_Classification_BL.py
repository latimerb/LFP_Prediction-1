import fastai
from fastai.text import *
from fastai.vision import *
from fastai.utils.mem import *
from fastai.basic_train import Learner, LearnerCallback
from fastai.tabular import *
import scipy.signal as s

path = Path("data/LFP_Prediction")

df = pd.read_csv(path/"model_LFP_and_FR.csv")

series = df["LFP"]

b_fit, a_fit = s.butter(2, [30, 80], 'bandpass', fs = 1000)

x = s.filtfilt(b_fit, a_fit, list(series))

mean = np.mean(x)

std = np.std(x)