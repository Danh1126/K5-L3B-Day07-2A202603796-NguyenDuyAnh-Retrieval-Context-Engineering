# Báo cáo Day 7

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang và số chunk:
+ Nguồn dữ liệu: Tài liệu lịch sử giáo dục History of Education - mã tài liệu gutenberg-27963, Project Gutenberg - thuộc phạm vi Public Domain
+ Quy mô corpus: 365 chunks được trích xuất từ 100+ trang nội dung.

- Vector store, embedding model, phiên bản và số chiều:
+ Vector store: Qdrant Local (qdrant-client, lưu trữ cục bộ tại data/day6_export/qdrant_day7_db, collection edu_history_C).
+ Embedding model: OpenAI text-embedding-3-small, số chiều vector 1536 dimensions, độ đo tương đồng Cosine similarity.

- Agent model, cách chạy và phân chia dev/test:
+ Agent LLM Controller: OpenAI gpt-4o-mini (temperature=0.0, response_format={"type": "json_object"}).
+ Kiến trúc chạy: Vòng lặp agent thực thi qua starter/agentic.py (run_agent) kết nối với adapter thật OpenAIController và QdrantDenseRetriever (src/real_agent.py). Không sử dụng dữ liệu gold hay mock fallback trong vòng lặp suy luận.
+ Phân chia benchmark (24 câu hỏi):
* Tập dev: 17 câu (15 câu có đáp án q01–q15 + 2 câu thiếu bằng chứng q21_unans, q22_unans) - dùng để thử nghiệm retrieval, phân tích lỗi và tinh chỉnh tham số.
* Tập test: 7 câu (5 câu có đáp án q16–q20 + 2 câu thiếu bằng chứng q23_unans, q24_unans) - giữ độc lập hoàn toàn, chỉ chạy đánh giá một lần sau khi chốt cấu hình trên dev.

- Cách xác định câu có/thiếu evidence và tiêu chí chấm:
+ Câu có evidence (answerable=True, 20 câu): Có ít nhất một đoạn trích dẫn nguyên văn (evidence_quote) gắn với source_id (gutenberg-27963) và gold_page cụ thể trong corpus; đã qua bước chuẩn hóa offset/page khớp 100% với chunks.json.

+ Câu thiếu evidence (answerable=False, 4 câu unans): Hỏi về các con số thống kê hoặc sự kiện giả định không tồn tại trong văn bản (ví dụ: tỷ lệ biết chữ năm 1750 ở Pháp, bằng tiến sĩ danh dự toán học của Roger Ascham), có nhãn gốc là INSUFFICIENT_EVIDENCE.

Tiêu chí chấm:
+ Grounded answer (0/1/2): 2 = trả lời đúng, đầy đủ ý và có citation hỗ trợ trực tiếp; 1 = đúng một phần hoặc citation chưa sát hoàn toàn; 0 = sai hoặc từ chối nhầm (abstain trên câu có đáp án).

+ Citation validity: Tỷ lệ citation [chunk-id] nằm trong danh sách context_ids đã truy xuất và nội dung chunk thực sự hỗ trợ phát biểu.

+ Abstention: Tỷ lệ từ chối trả lời chính xác (status == "abstain") trên nhóm câu answerable=False.

+ False abstention: Tỷ lệ từ chối nhầm trên nhóm câu answerable = True

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | TF-IDF lexical (starter/baseline.py), top_k=3, có metadata filter (chapter), không rerank, budget=800 ký tự | Mốc đối chứng từ khóa thuần túy với ngân sách context hẹp (800 ký tự) | dev | 7/15 (46.7%) | 10/15 (66.7%) | 4/15 (26.7) | Median 1.01 ms/câu (index 0.037 s), $0 |
| Phương pháp 2 | Dense Vector Retrieval (Qdrant text-embedding-3-small), top_k=3, có filter (chapter), không rerank, budget=1800 ký tự | Phương pháp mới 1 + Cải thiện budget: Embedding ngữ nghĩa bắt được câu hỏi diễn đạt đồng nghĩa; tăng budget lên 1800 ký tự để tránh cắt mất chunk ở hạng 2–3 | dev | 8/15 (53.3%) | 13/15 (86.7%) | 11/15 (73.3%) | Median 332.25 ms/câu, ~$0.00001/câu |
| Phương pháp 3 | Lexical BM25 (k1=1.5, b=0.75), top_k=3, có filter (chapter), không rerank, budget=1800 ký tự | Phương pháp mới 2: Khắc phục điểm yếu của Dense khi gặp tên riêng lịch sử (Loyola, Port Royalists) mà không tốn chi phí API embedding | dev | 9/15 (60.0%) | 11/15 (73.3%) | 10/15 (66.7%) | Median 3.22 ms/câu, $0 |
| Cải thiện 1 | Hybrid RRF mặc định (0.6 Dense + 0.4 BM25, rrf_k=60), top_k=3, có filter (chapter), RRF rank fusion, budget=1800 ký tự | Kết hợp điểm mạnh Hit@1 của BM25 (60.0%) và Recall@3 của Dense (86.7%) thông qua Reciprocal Rank Fusion | dev | 7/15 (46.7%) | 12/15 (80.0%) | 11/15 (73.3%) | Median 289.37 ms/câu, ~$0.00001/câu |
| Cải thiện 2 | Hybrid RRF tinh chỉnh + mở rộng Context (0.5 Dense + 0.5 BM25, rrf_k=30), top_k=5, có filter (chapter), budget=3000 ký tự | Giảm hằng số rrf_k từ 60 xuống 30 để tăng trọng số top đầu; mở rộng top_k=5 và budget=3000 ký tự để gom đủ các chunk liền kề cùng trang | dev | 7/15 (46.7%) | 12/15 (80.0%) (Recall@5: 12/15) | 11/15 (73.3%) | Median 355.68 ms/câu, ~$0.00001/câu |
| Cấu hình đã chọn | Dense Vector Retrieval (text-embedding-3-small), top_k=5, có filter (chapter), budget=3000 ký tự (tích hợp trong Agent max_searches=3) | Chốt Dense (đạt Recall@3 cao nhất 86.7% trên dev, không bị BM25 làm nhiễu hạng ở câu q02) kết hợp top_k=5, budget=3000 | test | — (chạy end-to-end qua Agent) | — (chạy end-to-end qua Agent) | 2/5 (40.0%) câu trả lời đạt đủ evidence | Median 3136.44 ms/câu (gồm LLM), $0.001723 toàn bộ 7 câu |
| Baseline | TF-IDF lexical (starter/baseline.py), top_k=3, có filter (chapter), budget=800 ký tự | Chạy đối chứng chuẩn hóa trên tập test | test | 3/5 (60.0%) | 5/5 (100.0%) | 2/5 (40.0%) | Median 3.38 ms/câu (index 0.074 s), $0 |

Một cấu hình có thể đồng thời là phương pháp mới và cải thiện; giải thích rõ. Báo `n/N` và tỷ lệ, ghi đơn vị thời gian/chi phí, so sánh test sau khi chốt. Nêu hai lỗi dev, nguyên nhân, cách thử và lý do chọn cấu hình:
- Quan hệ phương pháp mới và cải thiện: Dense Retrieval và BM25 là hai phương pháp truy xuất mới độc lập hoàn toàn với TF-IDF Baseline. Trong đó, việc chuyển từ TF-IDF (budget=800) sang Dense (budget=1800) vừa là phương pháp mới vừa là cải thiện lớn về Context Engineering (nâng Context Evidence Rate từ 26.7% lên 73.3%). Hybrid RRF là phương pháp thứ 3 kết hợp cả hai, đồng thời được thử nghiệm qua 2 biến thể tinh chỉnh tham số (Cải thiện 1 và Cải thiện 2).

Lỗi dev 1 — Trượt ghép cặp nguyên văn ở nhóm câu đa đoạn (q08, q11, q14):
+ Hiện tượng: Cả 4 cấu hình retrieval đều đạt Context Evidence Rate = 0 trên 3 câu q08, q11, q14 khi chạy 1-shot.
+ Nguyên nhân: Ở bước chia chunk Day 6, đoạn văn chứa đáp án bị tách thành 2 chunk liền kề trên cùng một trang; chunk đứng top-1 chỉ chứa nửa đầu của evidence_quote nên phép so khớp chuỗi nguyên văn (g['evidence'] in chunk['text']) báo trượt.
+ Cách thử & Lý do chọn cấu hình: Thử tăng top_k=5 và budget=3000 ký tự để đưa cả 2–3 chunk liền kề vào cùng một cửa sổ ngữ cảnh cho Agent tổng hợp.

Lỗi dev 2 — Nhiễu thứ hạng từ khóa ở câu diễn đạt đồng nghĩa (q02):
+ Hiện tượng: q02 được Dense Retrieval bắt trúng và đưa vào context thành công, nhưng bị trượt ở cả BM25 lẫn Hybrid RRF.
+ Nguyên nhân: Câu hỏi q02 sử dụng từ ngữ khái quát khác với từ vựng cổ trong sách thế kỷ XIX; BM25 chấm điểm cao nhầm cho các đoạn lặp lại từ khóa chung, kéo tụt hạng chunk đúng khi trộn RRF (0.6/0.4 và 0.5/0.5).
+ Lý do chọn cấu hình: Dựa trên kết quả dev, Dense Retrieval đạt Recall@3 = 13/15 (86.7%) và MRR = 0.6778 cao nhất, đồng thời không bị nhiễu từ khóa như BM25/Hybrid. Vì vậy, Dense Retrieval kết hợp top_k=5, budget=3000 được chốt làm retriever chính thức cho Agent trước khi chạy trên tập test.

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Agent Dense (top_k=5, budget=3000, max_searches=3) / test | 7/7 (100%) | Trung bình 1.0 lượt/câu (tối đa 3) | 0: 3, 1: 0, 2: 2 (N=5) — Mean: 0.80 / 2.00 | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 3/5 (60.0%) |

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q16 | Answerable (test) | search("Jesuit institutions students divided lower higher departments age admitted") trả về 5 chunks (budget=3000), trúng gutenberg-27963::C::p005::c000 | Students in Jesuit institutions were divided into two departments: the lower department (studia inferiora) and the higher department (studia superiora). Boys were admitted to the lower course at the age of fourteen... [gutenberg-27963::C::p005::c000]. | grade = 2, citation_support = [True] — Trả lời đầy đủ cả 2 vế (tên 2 phân khoa và độ tuổi nhập học 14), trích dẫn chính xác chunk nguồn. |
| q17 | Answerable (test) | search("La Salle three major contributions to education") kèm filter={"chapter": ...} trả về context thiếu đủ 3 đóng góp | Không đủ bằng chứng để trả lời. | grade = 0 (False abstention) — Từ chối nhầm do bộ lọc chapter quá chặt hoặc 3 đóng góp nằm rải rác ở các trang khác nhau chưa gom đủ trong 1 truy vấn. |
| q18 | Answerable (test) | search("Prussian government prohibit kindergartens 1851 how long") kèm filter={"chapter": ...} | Không đủ bằng chứng để trả lời. | grade = 0 (False abstention) — Từ chối nhầm; thông tin năm 1851 và thời hạn lệnh cấm bị lệch ranh giới filter chương trong metadata Day 6. |
| q19 | Answerable (test) | search("Horace Mann annual salary secretary Massachusetts State Board of Education") kèm filter={"chapter": ...} | Không đủ bằng chứng để trả lời. | grade = 0 (False abstention) — Từ chối nhầm; con số mức lương cụ thể nằm ở chunk liền kề không lọt top-5 dưới điều kiện filter cứng. |
| q20 | Answerable (test) | search("compulsory daily school attendance ages healthy child parents fail comply") trả về 5 chunks, trúng gutenberg-27963::C::p098::c001 | Daily school attendance is compulsory for every healthy child between the ages of six and fourteen. If parents fail to comply, they may be fined or imprisoned... [gutenberg-27963::C::p098::c001] | grade = 2, citation_support = [True] — Nêu chính xác độ tuổi (6–14) và hình phạt (phạt tiền, bỏ tù, đưa vào trường giáo dưỡng), citation chuẩn xác. |
| q23_unans | Thiếu bằng chứng (unans) | search("literacy percentage Jesuit schools France 1750") không có số liệu thống kê trong context | Không đủ bằng chứng để trả lời. | abstention_correct = True — Từ chối chính xác câu hỏi gài số liệu giả định không tồn tại trong sách. |
| q24_unans | Thiếu bằng chứng (unans) | search("Roger Ascham honorary doctorate mathematics The Scholemaster") không có chi tiết này trong context | Không đủ bằng chứng để trả lời. | abstention_correct = True — Từ chối chính xác, không suy diễn hay dùng kiến thức ngoài văn bản. |

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:
Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:
+ Traces (dev demo): starter/results/agent-dev-demo/traces.json
+ Báo cáo thử nghiệm retrieval (dev): starter/results/dev_retrieval_experiments.json
* Bộ artifact đánh giá chính thức (test):
+ Cases: starter/results/agent-test/cases.json
+ Traces: starter/results/agent-test/traces.json
+ Reviews: starter/results/agent-test/reviews.json
+ Summary metrics: starter/results/agent-test/summary.json
+ Lệnh chạy tái lập: python src/retrievers_and_experiments.py
python src/evaluate_test_and_report.py

* Thời gian và chi phí trên tập test (7 câu):
- Độ trễ trung vị (Median latency): 3136.44 ms/câu (dao động từ 2089.05 ms đến 4567.60 ms, bao gồm cả bước gọi embedding và suy luận LLM).
- Tổng token tiêu thụ: 9,151 prompt tokens + 579 completion tokens.
- Tổng chi phí ước tính: $0.001723 USD (trung bình ~$0.000246/câu), provider_errors = 0.

- Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:
+ So sánh trước/sau: So với TF-IDF Baseline ban đầu chỉ đạt Context_Hit = 4/15 (26.7%) do bị bó hẹp ở budget=800 ký tự và không có khả năng tự đánh giá bằng chứng, kiến trúc Agent kết hợp Qdrant Dense (top_k=5, budget=3000) đã nâng tỷ lệ đưa đúng bằng chứng vào context trên dev lên 73.3%, đạt 100% độ chính xác trích dẫn (2/2) và 100% abstention đúng (2/2) trên test với 0% hallucination.

+ Phân tích ví dụ lỗi (q17, q18, q19 — False Abstention): Trong 5 câu answerable của tập test, có 3 câu bị từ chối nhầm. Nguyên nhân chính đến từ sự kết hợp giữa bộ lọc metadata chapter cứng (nếu chunking Day 6 gán nhãn chương lệch 1–2 trang quanh ranh giới chương, Qdrant sẽ loại bỏ toàn bộ chunk mục tiêu) và prompt hệ thống ưu tiên an toàn cao khiến gpt-4o-mini chọn INSUFFICIENT_EVIDENCE ngay sau lượt tìm kiếm đầu tiên thay vì thử truy vấn lại không dùng filter hoặc mở rộng từ khóa.

+ Đánh đổi & Giới hạn:
+ Độ trễ & Chi phí vs. Chất lượng: Chuyển từ TF-IDF/BM25 (1–3 ms, $0) sang Dense + LLM Agent làm độ trễ tăng lên ~3.1 giây/câu và phát sinh chi phí API nhỏ ($0.0017 cho 7 câu), đổi lại có khả năng tổng hợp câu trả lời có trích dẫn chuẩn xác và biết từ chối khi thiếu dữ liệu.
+ Giới hạn hiện tại: Chưa áp dụng cơ chế fallback nới lỏng filter (tự động bỏ filters={"chapter": ...} nếu lượt tìm kiếm đầu tiên trả về ít kết quả hoặc model sắp abstention) và chưa có bước Cross-Encoder Reranking để đẩy các chunk nằm ở cuối trang lên top-3.

## Thử thách nâng cao (nếu thực hiện)

- Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:
- Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:
- Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | | | |
| Ảnh, sơ đồ | | | |
| Biểu đồ | | | |
| Multi-hop | | | |
| Text | | | |
| Thiếu bằng chứng | | | |
| **Toàn bộ** | | | |

- Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:
- Giới hạn, cách tái lập và đường dẫn benchmark:
