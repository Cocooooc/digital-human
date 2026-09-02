#download funasr
#conda create -n digital-human python=3.10 -y
#conda activate digital-human

from funasr import AutoModel

model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    device="cpu"
)

res = model.generate(
    input="test.wav"
)

print(res)