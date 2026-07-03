# Test results summary 

### General description 

Write something her.

### Model descriptions 

| model   | description                                                                                                                                                                                                                       |
|:--------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| v10     | ResNet 50 with one linear layer of size 32 for the body decoder and one linear layer of size 16 for the view decoder. Both have ReLU activation.                                                                                  |
| v11     | ResNet 50 with one linear layer of size 16 for the body decoder and one linear layer of size 8 for the view decoder. Both have ReLU activation.                                                                                   |
| v12     | ResNet 50 with one linear layer of size 64 for the body decoder and one linear layer of size 32 for the view decoder. Both have ReLU activation. This version has an added linear learning rate scheduler.                        |
| v13     | ResNet 50 with one linear layer of size 32 for the body decoder and one linear layer of size 16 for the view decoder. Both have ReLU activation. This version has an added linear learning rate scheduler.                        |
| v14     | ResNet 50 with one linear layer of size 64 for the body decoder and one linear layer of size 32 for the view decoder. Both have ReLU activation. This version has an added linear learning rate scheduler and has weighted loss.  |
| v15     | ResNet 50 with one linear layer of size 128 for the body decoder and one linear layer of size 32 for the view decoder. Both have ReLU activation. This version has an added linear learning rate scheduler and has weighted loss. |
| v2      | ResNet 50 with two linear layers of sizes 512 and 256 with ReLU activation for both decoders.                                                                                                                                     |
| v9      | ResNet 50 with one linear layer of size 64 for the body decoder and one linear layer of size 32 for the view decoder. Both have ReLU activation.                                                                                  |

### Model parameters 

| model                                                                                                           | model_version   | input_size    |   learning_rate |   train_batch_size |   max_epochs | imgnet_pretrained   | aug_color_jitter   | aug_rotation   |   lr_end_factor | body_loss_weight_path   | view_loss_weight_path   | sample_weight_path   | monitor       |
|:----------------------------------------------------------------------------------------------------------------|:----------------|:--------------|----------------:|-------------------:|-------------:|:--------------------|:-------------------|:---------------|----------------:|:------------------------|:------------------------|:---------------------|:--------------|
| Gen3_v1_vanilla_epoch63_04-09_00:35:56                                                                          | v1              | [3, 244, 244] |          0.0001 |                128 |          100 | False               | False              | False          |          nan    |                         |                         |                      | val/summed_f1 |
| Gen3_v1_imgnet_pretrained_epoch83_04-09_00:38:12                                                                | v1              | [3, 244, 244] |          0.0001 |                128 |          100 | True                | False              | False          |          nan    |                         |                         |                      | val/summed_f1 |
| Gen3_v1_imgnet_pretrained_color_jitter_epoch83_04-09_00:39:17                                                   | v1              | [3, 244, 244] |          0.0001 |                128 |          100 | True                | True               | False          |          nan    |                         |                         |                      | val/summed_f1 |
| Gen3_v1_imgnet_pretrained_color_jitter_rotation_epoch73_04-09_00:39:57                                          | v1              | [3, 244, 244] |          0.0001 |                128 |          100 | True                | True               | True           |          nan    |                         |                         |                      | val/summed_f1 |
| Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_epoch99_04-09_00:40:37                              | v1              | [3, 244, 244] |          0.0001 |                128 |          100 | True                | True               | True           |            0.01 |                         |                         |                      | val/summed_f1 |
| Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_oversampling_epoch99_04-09_00:41:18                 | v1              | [3, 244, 244] |          0.0001 |                128 |          100 | True                | True               | True           |            0.01 |                         |                         |                      | val/summed_f1 |
| Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_oversampling_loss_weighting_epoch_98_04-09_00:41:58 | v1              | [3, 244, 244] |          0.0001 |                128 |          100 | True                | True               | True           |            0.01 |                         |                         |                      | val/summed_f1 |
| Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_epoch91_04-10_00:35:56               | v1              | [3, 244, 244] |          0.0001 |                128 |          100 | True                | True               | True           |            0.01 |                         |                         |                      | val/summed_f1 |
| Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_epoch67_04-10_00:53:50               | v1              | [3, 244, 244] |          0.0001 |                128 |          100 | True                | True               | True           |            0.01 |                         |                         |                      | val/summed_f1 |
| Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_epoch85_04-10_00:51:06               | v1              | [3, 244, 244] |          0.0001 |                128 |          100 | True                | True               | True           |            0.01 |                         |                         |                      | val/summed_f1 |
| Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_epoch99_04-10_00:50:26               | v1              | [3, 244, 244] |          0.0001 |                128 |          100 | True                | True               | True           |            0.01 |                         |                         |                      | val/summed_f1 |
| Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting099_epoch95_04-11_00:46:02            | v1              | [3, 244, 244] |          0.0001 |                128 |          100 | True                | True               | True           |            0.01 |                         |                         |                      | val/summed_f1 |
| Gen3_v1_sgd_epoch_97_04-11_00:48:01                                                                             | v1              | [3, 244, 244] |          0.001  |                128 |          100 | False               | False              | False          |          nan    |                         |                         |                      | val/summed_f1 |

### Body part classification test results 

| model                                                                                                                                                    |   macro_f1 |   micro_f1 |   accuracy |   macro_precision |   macro_recall |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------|-----------:|-----------:|-----------:|------------------:|---------------:|
| /classification_models/Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_04-05_22:42:44/epoch99_summedf1_1.64.ckpt                             |     0.8611 |     0.9054 |     0.9054 |            0.8647 |         0.8594 |
| /classification_models/Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_04-09_01:13:01/epoch91_summedf1_1.63.ckpt              |     0.8591 |     0.9042 |     0.9042 |            0.8622 |         0.8580 |
| /classification_models/Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_04-09_01:13:01/epoch85_summedf1_1.63.ckpt              |     0.8590 |     0.9032 |     0.9032 |            0.8597 |         0.8592 |
| /classification_models/Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting099_04-10_11:45:51/epoch95_summedf1_1.62.ckpt           |     0.8589 |     0.9041 |     0.9041 |            0.8603 |         0.8582 |
| /classification_models/Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_04-09_01:13:01/epoch99_summedf1_1.62.ckpt              |     0.8556 |     0.9039 |     0.9039 |            0.8602 |         0.8536 |
| /classification_models/Gen3_v1_imgnet_pretrained_color_jitter_rotation_04-05_13:18:21/epoch73_summedf1_1.62.ckpt                                         |     0.8552 |     0.8988 |     0.8988 |            0.8611 |         0.8530 |
| /classification_models/Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_oversampling_loss_weighting_04-06_16:25:56/epoch98_summedf1_1.62.ckpt |     0.8538 |     0.9003 |     0.9003 |            0.8578 |         0.8520 |
| /classification_models/Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_04-09_01:13:01/epoch67_summedf1_1.63.ckpt              |     0.8536 |     0.9009 |     0.9009 |            0.8550 |         0.8535 |
| /classification_models/Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_oversampling_04-06_07:55:33/epoch99_summedf1_1.62.ckpt                |     0.8533 |     0.9008 |     0.9008 |            0.8567 |         0.8520 |
| /classification_models/Gen3_v1_imgnet_pretrained_color_jitter_04-04_23:55:23/epoch83_summedf1_1.59.ckpt                                                  |     0.8469 |     0.8961 |     0.8961 |            0.8512 |         0.8471 |
| /classification_models/Gen3_v1_imgnet_pretrained_04-04_11:16:42/epoch83_summedf1_1.61.ckpt                                                               |     0.8400 |     0.8996 |     0.8996 |            0.8436 |         0.8456 |
| /classification_models/Gen3_v1_vanilla_04-04_02:03:21/epoch63_summedf1_1.55.ckpt                                                                         |     0.8184 |     0.8752 |     0.8752 |            0.8152 |         0.8261 |
| /classification_models/Gen3_v1_sgd_04-10_02:31:54/epoch97_summedf1_1.52.ckpt                                                                             |     0.7942 |     0.8704 |     0.8704 |            0.7984 |         0.7934 |

### View classification test results 

| model                                                                                                                                                    |   macro_f1 |   micro_f1 |   accuracy |   macro_precision |   macro_recall |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------|-----------:|-----------:|-----------:|------------------:|---------------:|
| /classification_models/Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_04-05_22:42:44/epoch99_summedf1_1.64.ckpt                             |     0.8045 |     0.8685 |     0.8685 |            0.8209 |         0.7949 |
| /classification_models/Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting099_04-10_11:45:51/epoch95_summedf1_1.62.ckpt           |     0.8032 |     0.8666 |     0.8666 |            0.8199 |         0.7933 |
| /classification_models/Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_04-09_01:13:01/epoch67_summedf1_1.63.ckpt              |     0.8021 |     0.8656 |     0.8656 |            0.8172 |         0.7929 |
| /classification_models/Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_04-09_01:13:01/epoch85_summedf1_1.63.ckpt              |     0.7996 |     0.8637 |     0.8637 |            0.8133 |         0.7910 |
| /classification_models/Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_04-09_01:13:01/epoch91_summedf1_1.63.ckpt              |     0.7995 |     0.8639 |     0.8639 |            0.8128 |         0.7911 |
| /classification_models/Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_04-09_01:13:01/epoch99_summedf1_1.62.ckpt              |     0.7986 |     0.8643 |     0.8643 |            0.8142 |         0.7893 |
| /classification_models/Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_oversampling_04-06_07:55:33/epoch99_summedf1_1.62.ckpt                |     0.7960 |     0.8612 |     0.8612 |            0.8092 |         0.7878 |
| /classification_models/Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_oversampling_loss_weighting_04-06_16:25:56/epoch98_summedf1_1.62.ckpt |     0.7954 |     0.8606 |     0.8606 |            0.8098 |         0.7867 |
| /classification_models/Gen3_v1_imgnet_pretrained_04-04_11:16:42/epoch83_summedf1_1.61.ckpt                                                               |     0.7950 |     0.8583 |     0.8583 |            0.8062 |         0.7876 |
| /classification_models/Gen3_v1_imgnet_pretrained_color_jitter_rotation_04-05_13:18:21/epoch73_summedf1_1.62.ckpt                                         |     0.7945 |     0.8605 |     0.8605 |            0.8092 |         0.7857 |
| /classification_models/Gen3_v1_imgnet_pretrained_color_jitter_04-04_23:55:23/epoch83_summedf1_1.59.ckpt                                                  |     0.7873 |     0.8538 |     0.8538 |            0.7996 |         0.7796 |
| /classification_models/Gen3_v1_vanilla_04-04_02:03:21/epoch63_summedf1_1.55.ckpt                                                                         |     0.7663 |     0.8359 |     0.8359 |            0.7756 |         0.7604 |
| /classification_models/Gen3_v1_sgd_04-10_02:31:54/epoch97_summedf1_1.52.ckpt                                                                             |     0.7629 |     0.8387 |     0.8387 |            0.7913 |         0.7513 |

### Confusion matrices 

#### Gen3_v1_vanilla_epoch63_04-09_00:35:56 

![Image](Gen3_v1_vanilla_epoch63_04-09_00:35:56/conf_matrix_body.png)

![Image](Gen3_v1_vanilla_epoch63_04-09_00:35:56/conf_matrix_view.png)

#### Gen3_v1_imgnet_pretrained_epoch83_04-09_00:38:12 

![Image](Gen3_v1_imgnet_pretrained_epoch83_04-09_00:38:12/conf_matrix_body.png)

![Image](Gen3_v1_imgnet_pretrained_epoch83_04-09_00:38:12/conf_matrix_view.png)

#### Gen3_v1_imgnet_pretrained_color_jitter_epoch83_04-09_00:39:17 

![Image](Gen3_v1_imgnet_pretrained_color_jitter_epoch83_04-09_00:39:17/conf_matrix_body.png)

![Image](Gen3_v1_imgnet_pretrained_color_jitter_epoch83_04-09_00:39:17/conf_matrix_view.png)

#### Gen3_v1_imgnet_pretrained_color_jitter_rotation_epoch73_04-09_00:39:57 

![Image](Gen3_v1_imgnet_pretrained_color_jitter_rotation_epoch73_04-09_00:39:57/conf_matrix_body.png)

![Image](Gen3_v1_imgnet_pretrained_color_jitter_rotation_epoch73_04-09_00:39:57/conf_matrix_view.png)

#### Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_epoch99_04-09_00:40:37 

![Image](Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_epoch99_04-09_00:40:37/conf_matrix_body.png)

![Image](Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_epoch99_04-09_00:40:37/conf_matrix_view.png)

#### Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_oversampling_epoch99_04-09_00:41:18 

![Image](Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_oversampling_epoch99_04-09_00:41:18/conf_matrix_body.png)

![Image](Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_oversampling_epoch99_04-09_00:41:18/conf_matrix_view.png)

#### Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_oversampling_loss_weighting_epoch_98_04-09_00:41:58 

![Image](Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_oversampling_loss_weighting_epoch_98_04-09_00:41:58/conf_matrix_body.png)

![Image](Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_oversampling_loss_weighting_epoch_98_04-09_00:41:58/conf_matrix_view.png)

#### Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_epoch91_04-10_00:35:56 

![Image](Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_epoch91_04-10_00:35:56/conf_matrix_body.png)

![Image](Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_epoch91_04-10_00:35:56/conf_matrix_view.png)

#### Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_epoch67_04-10_00:53:50 

![Image](Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_epoch67_04-10_00:53:50/conf_matrix_body.png)

![Image](Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_epoch67_04-10_00:53:50/conf_matrix_view.png)

#### Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_epoch85_04-10_00:51:06 

![Image](Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_epoch85_04-10_00:51:06/conf_matrix_body.png)

![Image](Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_epoch85_04-10_00:51:06/conf_matrix_view.png)

#### Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_epoch99_04-10_00:50:26 

![Image](Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_epoch99_04-10_00:50:26/conf_matrix_body.png)

![Image](Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_epoch99_04-10_00:50:26/conf_matrix_view.png)

#### Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting099_epoch95_04-11_00:46:02 

![Image](Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting099_epoch95_04-11_00:46:02/conf_matrix_body.png)

![Image](Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting099_epoch95_04-11_00:46:02/conf_matrix_view.png)

#### Gen3_v1_sgd_epoch_97_04-11_00:48:01 

![Image](Gen3_v1_sgd_epoch_97_04-11_00:48:01/conf_matrix_body.png)

![Image](Gen3_v1_sgd_epoch_97_04-11_00:48:01/conf_matrix_view.png)



