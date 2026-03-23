
# 🚀 How to Update Hugging Face Dataset

## Method 1: Using Hugging Face CLI (Recommended)

1. **Install and Login**:
```bash
pip install huggingface_hub
huggingface-cli login
```

2. **Push Updated Dataset Card**:
```bash
cd /home/user/Eagle-Lander/DATA/huggingface-spikenaut-v2/dataset
huggingface-cli upload rmems/Spikenaut-SNN-v2-Telemetry-Data-Weights-Parameters dataset_card.json --commit-message="🦁 MASSIVE ENHANCEMENT v2.1: Complete neuromorphic blockchain ecosystem (635MB, 1.4M+ records)"
```

3. **Push Additional Data Files**:
```bash
# Push training data
huggingface-cli upload rmems/Spikenaut-SNN-v2-Telemetry-Data-Weights-Parameters training/ --folder training/ --commit-message="Add SNN training data (40K+ records)"

# Push mining data
huggingface-cli upload rmems/Spikenaut-SNN-v2-Telemetry-Data-Weights-Parameters mining/ --folder mining/ --commit-message="Add mining operation logs (55MB)"

# Push operations data
huggingface-cli upload rmems/Spikenaut-SNN-v2-Telemetry-Data-Weights-Parameters operations/ --folder operations/ --commit-message="Add system monitoring telemetry"

# Push research data
huggingface-cli upload rmems/Spikenaut-SNN-v2-Telemetry-Data-Weights-Parameters research/ --folder research/ --commit-message="Add neuromorphic research dataset (380MB)"
```

## Method 2: Using Python API

```python
from huggingface_hub import HfApi, Repository
import json

# Login and upload
api = HfApi()
api.upload_file(
    path_or_fileobj="dataset_card.json",
    path_in_repo="dataset_card.json",
    repo_id="rmems/Spikenaut-SNN-v2-Telemetry-Data-Weights-Parameters",
    repo_type="dataset",
    commit_message="🦁 MASSIVE ENHANCEMENT v2.1: Complete neuromorphic blockchain ecosystem"
)
```

## Method 3: Manual Upload

1. Go to: https://huggingface.co/datasets/rmems/Spikenaut-SNN-v2-Telemetry-Data-Weights-Parameters
2. Click "Edit dataset card"
3. Copy the content from `README_V2.1_UPDATE.md`
4. Update the dataset card with the enhanced information
5. Upload additional files using the web interface

---

## 📊 What Will Be Updated

### **Dataset Card Changes**:
- ✅ Pretty name: "Complete Neuromorphic Blockchain Ecosystem"
- ✅ Description: Massive enhancement alert with 635MB details
- ✅ Tags: 25 comprehensive tags for discoverability
- ✅ Size categories: Multiple categories for 1.4M+ records
- ✅ Task categories: 5 specialized task categories
- ✅ Version: 2.1.0 (massive enhancement)

### **Additional Data**:
- ✅ Training data folder with SNN spike patterns
- ✅ Mining data folder with BzMiner operation logs
- ✅ Operations data folder with system monitoring
- ✅ Research data folder with neuromorphic dataset

---

## 🎯 Expected Results

After updating, your dataset will show:
- **635MB total size** (vs ~200MB before)
- **5 data collections** (vs 1 before)
- **1.4M+ records** (vs 8 before)
- **Complete ecosystem** positioning
- **Professional discoverability** across multiple communities

---

## 🦁 Ready to Upload!

Your enhanced dataset is ready to become the world's most comprehensive neuromorphic blockchain dataset!
