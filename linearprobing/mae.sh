./runner python3 main/clf_tasks_main.py \
    --runners 4 \
    -- \
    --seed 0 1 2 3 4 \
    --init_weights \
        _weights/v3_multimae-b_pret-multimae_49_1600e_bscan_512-32_checkpoint-1599.pth \
    --ft_weights \
        _weights/best-model_MAE0.07Falselr0.0001batch_size128LossesAll_bscan_multimae.pth \
    --linear_probing true \
    --batch_size 0 \
    --representation_method concat \
    --nb_classes 0 \
    --early_stopping_epochs 20 \
    --early_start_from 20 \
    --val_metric bacc \
    --early_stopping_delta 0.001 \
    --val_metric_two loss \
    --early_stopping_delta_two 0.001 \
    --epochs 1000 \
    --version finetunedMAEOCT \
    --fill 0 \
    --data_set \
        OCTID OCTDL Noor_Eye_Hospital Duke_iAMD GAMMAv2 Harvard_Glaucoma 9C2D POAGD2D NEHUT2D
exit
