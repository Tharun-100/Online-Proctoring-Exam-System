# AI-Based Fraud Detection System for Online Proctored Exams

## 1. INTRODUCTION

### 1.1 Context and Problem Statement

The widespread adoption of online education has fundamentally transformed the accessibility and scalability of learning systems, enabling institutions to serve millions of students across geographic boundaries. However, this digital transformation has introduced unprecedented challenges to academic integrity and assessment validity.[67][74] The proliferation of online examinations has exposed critical vulnerabilities in traditional invigilation practices, as the absence of physical supervision creates an environment susceptible to diverse forms of academic misconduct.

Research indicates that between 50% and 70% of students admit to engaging in cheating behaviors in traditional educational settings,[70] with online assessments presenting even more acute challenges.[67] This problem has been dramatically exacerbated in 2025 by the rapid proliferation of generative artificial intelligence tools. Current data reveals that 92% of students utilize AI in some form (compared to 66% in 2024), and a staggering 88% employed generative AI tools to complete assessments in the 2024-25 academic year.[64][61] Additionally, 62% of students have explicitly attempted to use ChatGPT or similar tools during online examinations, circumventing traditional detection methods.[68] These statistics underscore the fundamental inadequacy of conventional integrity safeguards in an era of accessible, powerful generative models.

Device-based cheating compounds these challenges substantially. Commercial proctoring providers and institutional investigations reveal that approximately 71% of detected online proctored examination violations involve secondary devices such as smartphones, smartwatches, or hidden communication interfaces.[62][65] These vulnerabilities highlight a critical gap between the technological sophistication of current assessment systems and the evolving tactics employed by academically dishonest students.

Traditional human proctoring, while educationally sound, faces severe scalability and economic constraints. Manual invigilation typically requires approximately 4-5 proctors per 20-30 candidates and incurs costs between $5 and $15 per examination, making it economically infeasible for large-scale educational deployments.[16][25] In contrast, AI-based proctoring platforms can monitor thousands of candidates simultaneously at a fraction of the per-exam cost.[25] Consequently, institutions urgently require automated, scalable solutions that maintain assessment integrity without proportional increases in human resources or operational expenditure.[22][52]

### 1.2 Aim

The primary aim of this research is to develop and evaluate an artificial intelligence-based fraud detection system capable of automatically identifying suspicious behaviors during online proctored examinations through real-time analysis of student behavioral features extracted via computer vision and machine learning. Recent research emphasizes the importance of multi-modal approaches that combine facial analysis, hand movement detection, and device detection to capture a broader spectrum of cheating behaviors.[19][44][40]

Specifically, this system aims to:

- Create an efficient, scalable solution for detecting academic misconduct in remote examination environments that operates at negligible cost per examination, enabling equitable deployment across diverse institutional contexts
- Implement a multi-modal feature extraction pipeline utilizing MediaPipe-based facial and hand landmark detection combined with YOLOv8 object detection to capture comprehensive behavioral indicators[42][45][46]
- Develop a robust machine learning classification model employing XGBoost for binary classification (suspicious vs. normal behavior) that achieves performance targets exceeding 90% accuracy while maintaining balanced precision and recall[21][24][33]
- Provide institutional stakeholders with an interpretable, audit-trail fraud detection mechanism that generates visual evidence and confidence scores suitable for investigation and decision-making
- Establish a technical foundation for full-stack deployment systems that integrate real-time detection with comprehensive exam data analytics and institutional administrative systems

### 1.3 Objectives

The specific technical and analytical objectives guiding this project include:

**Objective 1: Multi-Modal Feature Extraction Pipeline.** To implement a comprehensive feature extraction pipeline leveraging MediaPipe face landmark detection (468 facial keypoints), MediaPipe hand landmark detection (21 hand keypoints per hand), and YOLOv8 object detection to identify prohibited items, secondary devices, and environmental anomalies.[42][45][46][48] MediaPipe solutions provide real-time inference suitable for both offline image analysis and live video streams, while YOLOv8 offers computational efficiency optimized for mobile and edge deployment scenarios.[46][48] Extracted features will be processed and normalized to generate a structured feature representation suitable for machine learning classification.

**Objective 2: Suspicious Behavior Classification Model.** To train and optimize an XGBoost classifier capable of binary classification (suspicious vs. normal examination behavior) with performance targets of >90% accuracy, >85% precision, and >85% recall across balanced test sets. XGBoost has demonstrated strong performance on tabular feature data and robustness to noisy labels in security and anomaly detection applications,[21][24][33] making it well-suited to this classification task. The model will be trained on the student suspicious behaviors detection dataset_V1 with rigorous cross-validation and hyperparameter optimization protocols.

**Objective 3: Single-Image Prediction System.** To develop a practical proof-of-concept application accepting uploaded examination images and producing real-time binary predictions (suspicious/normal behavior) with confidence scores, probability estimates, and visual highlighting of features driving classification decisions. This system will provide educators and proctoring staff with immediate, actionable intelligence for examination monitoring and incident investigation.

**Objective 4: Comprehensive Evaluation Framework.** To establish rigorous evaluation methodologies employing stratified test sets derived from the student suspicious behaviors detection dataset_V1, including performance metrics such as accuracy, precision, recall, F1-score, area under the receiver operating characteristic curve (AUROC), and confusion matrices. Evaluation will assess model performance across diverse behavioral categories, demographic subgroups, and image quality conditions to identify potential bias or performance disparities.

**Objective 5: Scalability and Real-Time Performance.** To design the system architecture for efficient inference on resource-constrained environments, including mobile devices and edge computing platforms, ensuring practical deployment feasibility across institutional infrastructure. Model quantization, optimization, and lightweight architecture selection will be prioritized to enable sub-second inference latency suitable for real-time monitoring applications.

### 1.4 Wider Purpose and Broader Impact

Beyond the immediate technical objectives of detecting examination fraud, this project serves a broader societal purpose centered on preserving the foundational integrity of educational assessment in an era of digital transformation and artificial intelligence proliferation.

**Educational Integrity as Social Infrastructure.** Academic assessment serves as a critical social contract between educational institutions, employers, and society at large. The validity and fairness of examinations directly determine the credibility of academic credentials and, by extension, the professional qualifications of graduates.[67][70] When examination integrity is compromised through unchecked cheating, the entire educational ecosystem experiences cascading negative consequences: institutions lose institutional credibility, employers cannot reliably differentiate competent candidates from dishonest ones, and honest students face devaluation of their achievements through credential inflation.[74] By enabling scalable, automated fraud detection, this system helps restore faith in the credibility of online assessments and ensures that academic credentials maintain meaningful predictive validity regarding candidate competence.

**Equitable Access and Cost Democratization.** Traditional human proctoring creates significant barriers to educational access, particularly for institutions in resource-constrained regions or serving underserved populations. The cost differential between human proctoring ($5-15 per exam) and AI-based solutions (<$1 per exam) represents a 10-fold reduction in operational expense.[16][25] This economic efficiency enables institutions serving low-income students, marginalized communities, and developing nations to implement sophisticated fraud detection without resorting to compromised assessment methodologies. By reducing the financial burden of maintaining assessment integrity, this technology democratizes access to credible online education and enables equitable scale-out of remote learning programs.[22]

**Technological Response to Evolving Threats.** The emergence of large language models and generative AI has fundamentally altered the threat landscape in academic assessment. Traditional anti-cheating measures—designed to counter discrete, observable behaviors like consulting unauthorized materials or receiving external assistance—prove inadequate against generative AI tools that operate invisibly within the digital examination environment.[61][64][68] An AI-based fraud detection system combining behavioral analysis with technological threat detection represents the only proportional response to this escalating threat. This project contributes to the broader effort of developing educational systems that can evolve alongside emerging technologies and maintain credibility in an AI-augmented world.

**Workforce Development and Professional Credibility.** In professional domains where competence directly affects public safety and client welfare—such as medicine, engineering, law, and aviation—examination integrity is not merely an institutional concern but a public health and safety imperative.[70] By enabling scalable verification of authentic competence in online assessment contexts, this system contributes to maintaining professional credibility and public trust in regulated professions. The ability to confidently certify professional competence through remote assessment mechanisms expands access to professional credentialing while safeguarding public welfare.

**Institutional Scalability for Educational Transformation.** The COVID-19 pandemic permanently transformed educational delivery models, with hybrid and fully remote programs now constituting a permanent segment of higher education. Institutions managing thousands of concurrent online examinations cannot reasonably employ proportional increases in human proctoring staff.[16][25] An AI-based fraud detection system enables institutions to scale assessment capacity while maintaining integrity standards—a capability essential for the future of accessible, affordable, and scalable higher education.[22][52] This technological capability represents a prerequisite for equitable global access to credible educational assessment.

---

## REFERENCES

[16] Think Proctor. (2025). How Virtual Proctoring Detects and Prevents Fraud in Online Exams. Retrieved from https://thinkexam.com/blog/how-virtual-proctoring-detects-and-prevents-fraud-in-online-exams/

[19] ArXiv. (2024). AutoOEP - A Multi-modal Framework for Online Exam Proctoring. Retrieved from https://arxiv.org/html/2509.10887v1

[21] DPSS INESC-ID. (2025). Network Intrusion Detection with XGBoost. Retrieved from https://www.dpss.inesc-id.pt/~mpc/pubs/XGBoost_chapter.pdf

[22] IJSRA. (2025). Online Exam Proctoring Application Using AI. Retrieved from https://journalijsra.com/sites/default/files/fulltext_pdf/IJSRA-2025-1440.pdf

[24] WARSE. (2020). XGBoost Classification Based Network Intrusion Detection System. Retrieved from http://www.warse.org/IJATCSE/static/pdf/file/ijatcse55912020.pdf

[25] Incruiter. (2025). AI Proctoring Software for Secure Online Assessments & Interviews. Retrieved from https://incruiter.com/blog/ai-proctoring-software-online-assessments-interviews/

[33] ScienceDirect. (2025). Data Adjusting Strategy and Optimized XGBoost Algorithm. Retrieved from https://www.sciencedirect.com/science/article/abs/pii/S0016003223005586

[40] Journal of Educational Services Research. (2024). Deep Learning-Based Multimodal Cheating Detection in Online Proctored Exams. Retrieved from https://journal.esrgroups.org/jes/article/view/7480

[42] IJARIIE. (2025). Hand and Face Landmarks Detection Using Media Pipe. Retrieved from https://ijariie.com/AdminUploadPdf/HAND_AND_FACE_LANDMARKS_DETECTION_USING_MEDIA_PIPE_AND_AI_ijariie26424.pdf

[44] PMC. (2023). An Automated Online Proctoring System Using Attentive-Net. Retrieved from https://pmc.ncbi.nlm.nih.gov/articles/PMC9944407/

[45] Google AI Edge. (2024). MediaPipe Hands Documentation. Retrieved from https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

[46] GeeksforGeeks. (2024). Object Detection using YOLOv8. Retrieved from https://www.geeksforgeeks.org/machine-learning/object-detection-using-yolov8/

[48] Hugging Face. (2024). Qualcomm YOLOv8-Detection. Retrieved from https://huggingface.co/qualcomm/YOLOv8-Detection

[52] IJARSCT. (2025). Online Exam Proctoring System Using ML. Retrieved from https://ijarsct.co.in/Paper11649.pdf

[61] ArtiSmart AI. (2025). AI Plagiarism Statistics: Navigating Academic Integrity in the Age of AI. Retrieved from https://artsmart.ai/blog/ai-plagiarism-statistics/

[62] Honorlock. (2025). 13 Ways to Prevent Cheating on Online Tests. Retrieved from https://honorlock.com/blog/4-ways-to-prevent-cheating-on-online-exams/

[64] Meazure Learning. (2025). By the Numbers: Academic Integrity in Higher Education. Retrieved from https://www.meazurelearning.com/resources/by-the-numbers-academic-integrity-in-higher-education

[65] Assess.com. (2025). Tips to Catch Student Cheating in Online Proctored Exams. Retrieved from https://assess.com/remote-proctoringsecurity/

[67] Frontiers in Education. (2021). Academic Integrity in Online Assessment: A Research Review. Retrieved from https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2021.639814/full

[68] Eklavvya. (2025). How to Prevent Cheating in Online Exams: 15 Proven Methods. Retrieved from https://www.eklavvya.com/blog/prevent-cheating-online-exams/

[70] ICAI. (2025). Facts & Statistics on Academic Integrity. Retrieved from https://academicintegrity.org/aws/ICAI/pt/sp/facts

[74] Frontiers in Education. (2022). A Systematic Review of Research on Cheating in Online Exams. Retrieved from https://pmc.ncbi.nlm.nih.gov/articles/PMC8898996/
