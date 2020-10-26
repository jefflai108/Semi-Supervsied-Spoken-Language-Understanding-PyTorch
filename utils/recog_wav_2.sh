#!/bin/bash

# Copyright 2019 Nagoya University (Takenori Yoshimura)
#           2019 RevComm Inc. (Takekatsu Hiramura)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

if [ ! -f path.sh ] || [ ! -f cmd.sh ]; then
    echo "Please change current directory to recipe directory e.g., egs/tedlium2/asr1"
    exit 1
fi

. ./path.sh

# general configuration
backend=pytorch
stage=3        # start from 0 if you need to start from data preparation
stop_stage=3
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=10
debugmode=1
verbose=1      # verbose option

# feature configuration
do_delta=false
cmvn=

# rnnlm related
use_lang_model=true
lang_model=

# decoding parameter
recog_model=
decode_config=
decode_dir=decode
api=v2

# download related
models=tedlium2.transformer.v1

help_message=$(cat <<EOF
Usage:
    $0 [options] <wav_file>

Options:
    --backend <chainer|pytorch>     # chainer or pytorch (Default: pytorch)
    --ngpu <ngpu>                   # Number of GPUs (Default: 0)
    --decode_dir <directory_name>   # Name of directory to store decoding temporary data
    --models <model_name>           # Model name (e.g. tedlium2.transformer.v1)
    --cmvn <path>                   # Location of cmvn.ark
    --lang_model <path>             # Location of language model
    --recog_model <path>            # Location of E2E model
    --decode_config <path>          # Location of configuration file
    --api <api_version>             # API version (v1 or v2, available in only pytorch backend)

Example:
    # Record audio from microphone input as example.wav
    rec -c 1 -r 16000 example.wav trim 0 5

    # Decode using model name
    $0 --models tedlium2.transformer.v1 example.wav

    # Decode with streaming mode (only RNN with API v1 is supported)
    $0 --models tedlium2.rnn.v2 --api v1 example.wav

    # Decode using model file
    $0 --cmvn cmvn.ark --lang_model rnnlm.model.best --recog_model model.acc.best --decode_config conf/decode.yaml example.wav

    # Decode with GPU (require batchsize > 0 in configuration file)
    $0 --ngpu 1 example.wav

Available models:
    - tedlium2.rnn.v1
    - tedlium2.rnn.v2
    - tedlium2.transformer.v1
    - tedlium3.transformer.v1
    - librispeech.transformer.v1
    - librispeech.transformer.v1.transformerlm.v1
    - commonvoice.transformer.v1
    - csj.transformer.v1
EOF
)
. utils/parse_options.sh || exit 1;

# make shellcheck happy
train_cmd=
decode_cmd=

. ./cmd.sh

wav_dir=$1
download_dir=${decode_dir}/download

if [ $# -lt 1 ]; then
    echo "${help_message}"
    exit 1;
fi

set -e
set -u
set -o pipefail

# check api version
if [ "${api}" = "v2" ] && [ "${backend}" = "chainer" ]; then
    echo "chainer backend does not support api v2." >&2
    exit 1;
fi

# Check model name or model file is set
if [ -z $models ]; then
    if [ $use_lang_model = "true" ]; then
        if [[ -z $cmvn || -z $lang_model || -z $recog_model || -z $decode_config ]]; then
            echo 'Error: models or set of cmvn, lang_model, recog_model and decode_config are required.' >&2
            exit 1
        fi
    else
        if [[ -z $cmvn || -z $recog_model || -z $decode_config ]]; then
            echo 'Error: models or set of cmvn, recog_model and decode_config are required.' >&2
            exit 1
        fi
    fi
fi

dir=${download_dir}/${models}
mkdir -p ${dir}

function download_models () {
    if [ -z $models ]; then
        return
    fi

    file_ext="tar.gz"
    case "${models}" in
        "tedlium2.rnn.v1") share_url="https://drive.google.com/open?id=1UqIY6WJMZ4sxNxSugUqp3mrGb3j6h7xe" ;;
        "tedlium2.rnn.v2") share_url="https://drive.google.com/open?id=1cac5Uc09lJrCYfWkLQsF8eapQcxZnYdf" ;;
        "tedlium2.transformer.v1") share_url="https://drive.google.com/open?id=1heuP2G5YX5u4hERs370eF-1MG2DT50zR" ;;
        "tedlium3.transformer.v1") share_url="https://drive.google.com/open?id=1ESVWQp0ZMhenF_Dt1n47suMK8NJ8hm0A" ; file_ext="zip" ;;
        "librispeech.transformer.v1") share_url="https://drive.google.com/open?id=1BtQvAnsFvVi-dp_qsaFP7n4A_5cwnlR6" ;;
        "librispeech.transformer.v1.transformerlm.v1") share_url="https://drive.google.com/open?id=17cOOSHHMKI82e1MXj4r2ig8gpGCRmG2p" ;;
        "commonvoice.transformer.v1") share_url="https://drive.google.com/open?id=1tWccl6aYU67kbtkm8jv5H6xayqg1rzjh" ;;
        "csj.transformer.v1") share_url="https://drive.google.com/open?id=120nUQcSsKeY5dpyMWw_kI33ooMRGT2uF" ;;
        *) echo "No such models: ${models}"; exit 1 ;;
    esac

    if [ ! -e ${dir}/.complete ]; then
        download_from_google_drive.sh ${share_url} ${dir} ${file_ext}
        touch ${dir}/.complete
    fi
}

# Download trained models
if [ -z "${cmvn}" ]; then
    download_models
    cmvn=$(find ${download_dir}/${models} -name "cmvn.ark" | head -n 1)
fi
if [ -z "${lang_model}" ] && ${use_lang_model}; then
    download_models
    lang_model=$(find ${download_dir}/${models} -name "rnnlm*.best*" | head -n 1)
fi
if [ -z "${recog_model}" ]; then
    download_models
    recog_model=$(find ${download_dir}/${models} -name "model*.best*" | head -n 1)
fi
if [ -z "${decode_config}" ]; then
    download_models
    decode_config=$(find ${download_dir}/${models} -name "decode*.yaml" | head -n 1)
fi

# Check file existence
if [ ! -f "${cmvn}" ]; then
    echo "No such CMVN file: ${cmvn}"
    exit 1
fi
if [ ! -f "${lang_model}" ] && ${use_lang_model}; then
    echo "No such language model: ${lang_model}"
    exit 1
fi
if [ ! -f "${recog_model}" ]; then
    echo "No such E2E model: ${recog_model}"
    exit 1
fi
if [ ! -f "${decode_config}" ]; then
    echo "No such config file: ${decode_config}"
    exit 1
fi
if [ ! -d "${wav_dir}" ]; then
    echo "No such wav dir: ${wav_dir}"
    exit 1
fi

base=$(basename $wav_dir)
decode_dir=${decode_dir}/${base}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"

    utils/utt2spk_to_spk2utt.pl <$wav_dir/utt2spk >$wav_dir/spk2utt

    mkdir -p ${decode_dir}/data
    cp $wav_dir/wav.scp ${decode_dir}/data/wav.scp
    cp $wav_dir/utt2spk ${decode_dir}/data/utt2spk
    cp $wav_dir/text ${decode_dir}/data/text
    cp $wav_dir/spk2utt ${decode_dir}/data/spk2utt
    utils/fix_data_dir.sh ${decode_dir}/data
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"

    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
        ${decode_dir}/data ${decode_dir}/log ${decode_dir}/fbank

    feat_recog_dir=${decode_dir}/dump; mkdir -p ${feat_recog_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        ${decode_dir}/data/feats.scp ${cmvn} ${decode_dir}/log \
        ${feat_recog_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Json Data Preparation"

    dict=${decode_dir}/dict
    echo "<unk> 1" > ${dict}
    feat_recog_dir=${decode_dir}/dump
    data2json.sh --feat ${feat_recog_dir}/feats.scp \
        ${decode_dir}/data ${dict} > ${feat_recog_dir}/data.json
    rm -f ${dict}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Decoding"
    if ${use_lang_model}; then
        recog_opts="--rnnlm ${lang_model}"
    else
        recog_opts=""
    fi
    feat_recog_dir=${decode_dir}/dump

    # split data
    splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

    ${decode_cmd} JOB=1:${nj} ${decode_dir}/log/decode.JOB.log \
        asr_recog.py \
        --config ${decode_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --debugmode ${debugmode} \
        --verbose ${verbose} \
        --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
        --result-label ${feat_recog_dir}/split${nj}utt/result.JOB.json \
        --model ${recog_model} \
        --api ${api} \
        ${recog_opts}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: results"
    echo ""

    concatjson.py ${decode_dir}/dump/split${nj}utt/result.*.json > ${decode_dir}/result.json

    recog_text=$(grep rec_text ${decode_dir}/result.json | sed -e 's/.*: "\(.*\)".*/\1/' | sed -e 's/<eos>//')
    echo "Recognized text: ${recog_text}"
    echo ""

    json2trn_wo_dict.py ${decode_dir}/result.json --num-spkrs 1 --refs ${decode_dir}/ref_org.wrd.trn --hyps ${decode_dir}/hyp_org.wrd.trn

    cat ${decode_dir}/hyp_org.wrd.trn | sed -e 's/▁//' | sed -e 's/▁/ /g' > ${decode_dir}/hyp.wrd.trn
    cat ${decode_dir}/ref_org.wrd.trn | sed -e 's/\.//g' -e 's/\,//g' > ${decode_dir}/ref.wrd.trn

    cat ${decode_dir}/hyp.wrd.trn | awk -v FS='' '{a=0;for(i=1;i<=NF;i++){if($i=="("){a=1};if(a==0){printf("%s ",$i)}else{printf("%s",$i)}}printf("\n")}' > ${decode_dir}/hyp.trn
    cat ${decode_dir}/ref.wrd.trn | awk -v FS='' '{a=0;for(i=1;i<=NF;i++){if($i=="("){a=1};if(a==0){printf("%s ",$i)}else{printf("%s",$i)}}printf("\n")}' > ${decode_dir}/ref.trn

    sclite -r ${decode_dir}/ref.trn trn -h ${decode_dir}/hyp.trn -i rm -o all stdout > ${decode_dir}/results.md
    echo "write a CER result in ${decode_dir}/results.md"
    grep -e Avg -e SPKR -m 2 ${decode_dir}/results.md

    sclite -r ${decode_dir}/ref.wrd.trn trn -h ${decode_dir}/hyp.wrd.trn -i rm -o all stdout > ${decode_dir}/results.wrd.md
    echo "write a WER result in ${decode_dir}/results.wrd.md"
    grep -e Avg -e SPKR -m 2 ${decode_dir}/results.wrd.md

    sclite -r ${decode_dir}/ref_org.wrd.trn trn -h ${decode_dir}/hyp.wrd.trn trn -i rm -o all stdout > ${decode_dir}/results_w_punc.wrd.md
    echo "write a WER result in ${decode_dir}/results_w_punc.wrd.md"
    grep -e Avg -e SPKR -m 2 ${decode_dir}/results_w_punc.wrd.md

fi
