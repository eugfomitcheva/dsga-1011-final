python main_candidate_generation.py \
--dataset xsum \
--model_type pegasus \
--model google/pegasus-large \
--model_name pegasus_unsupervised \
--cache_dir ../../../hf_models/pegasus-large \
--load_model False \
--val_dataset val \
--inference_bs 2 \
--save_summaries True \
--generation_method diverse_beam_search \
--num_return_sequences 15 \
--num_beams 15 \
--num_beam_groups 15 \
