# Hướng dẫn dự án TRAMIF

Repo này tập hợp các bước phát triển thí nghiệm và pipeline dùng để tạo kết quả cuối. Đọc [protocol của bản nộp](RESEARCH_PROTOCOL_V1.md), [kết quả Future](FINAL_FUTURE_RESULTS.md) và [hướng dẫn tái lập](REPRODUCIBILITY.md) để theo dõi các điều kiện đánh giá.

| Công đoạn | Mã và tài liệu |
| --- | --- |
| Audit dữ liệu và xây dựng manifest | [Audit BODMAS của nhóm](../scripts/audit_bodmas_metadata.py), [manifest theo thời gian](../scripts/build_temporal_manifest.py), [quy tắc cohort và audit bản nộp](validation_audit.md) |
| Tạo ảnh đa view | [Quy trình dataset đa view của nhóm](TRAMIF_DISTRIBUTED_MULTIVIEW_V2.md), [đặc tả preprocessing bản nộp](image_preprocessing_spec.md) |
| Huấn luyện | [Quy trình huấn luyện theo thời gian của nhóm](TRAMIF_TEMPORAL_TRAINING.md), [trainer cho các seed của bản nộp](../scripts/train_final_seed.py), [thiết lập thí nghiệm cuối](FINAL_MULTISEED_EXECUTION.md) |
| Calibration và fusion | [Pipeline đánh giá lịch sử của nhóm](TRAMIF_PREDICTION_PIPELINE.md), [uniform fusion theo bộ artifact đã ghim](../scripts/evaluate_uniform_fusion_from_register.py), [tích hợp và khóa D060](D060_PRE_FUTURE_INTEGRATION_RESULTS.md) |
| Đánh giá Future | [Chuẩn bị logits D061](D061_PROTECTED_FUTURE_INPUT_RESULTS.md), [kết quả D062](FINAL_FUTURE_RESULTS.md), [script chấm điểm](../scripts/score_final_future.py) |

Pipeline phát triển của nhóm hướng dẫn seeds `11,22,33,44,55`. Kết quả cuối theo protocol đã khóa dùng seeds `42–46`. Mã và hồ sơ cho pipeline kết quả cuối được bổ sung từ [Wakeupat5am/TRAMIF tại 410675f](https://github.com/Wakeupat5am/TRAMIF/commit/410675f8418acfac882e496530c4508ad381f773).

Tái lập số liệu D062 cần các artifact đã khóa của D060 và D061 theo [hợp đồng tái lập](REPRODUCIBILITY.md).
