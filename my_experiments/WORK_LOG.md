# Work Log - December 7, 2025

## Progress Summary
1. **Environment Setup**: 
   - Verified Python 3.12, PyTorch 2.7.0, and CUDA 12.6 environment.
   - Confirmed `sam3` package installation.

2. **Codebase Preparation**:
   - Resolved Git conflicts in `sam3/model/sam3_tracker_base.py` and `sam3/model/sam3_tracking_predictor.py`.
   - Created `my_experiments/` directory for isolated testing.

3. **Script Development**:
   - `my_experiments/inference_test.py`: Main script for testing SAM3 image inference. Configured to download weights from Hugging Face.
   - `my_experiments/auth_hf.py`: Helper script for Hugging Face authentication.
   - `my_experiments/debug_download.py`: Diagnostic script that confirmed the 403 Forbidden error.

4. **Issue Diagnosis**:
   - Encountered `403 Forbidden` when attempting to download `facebook/sam3`.
   - **Cause**: The Hugging Face token used lacked permissions for "Gated Repositories".
   - **Solution Identified**: Need to use a **Classic Token** (Read) or a **Fine-grained Token** with "Public Gated Repositories" permission.

## Next Steps (To Do)
1. **Update Token**:
   - Run `python my_experiments/auth_hf.py` and input a new, valid token.
   
2. **Verify Inference**:
   - Run `python my_experiments/inference_test.py`.
   - Confirm that `sam3.pt` downloads successfully.
   - Check `my_experiments/output/inference_log.txt` and `prediction.json` for results.

3. **Documentation**:
   - Continue updating `my_experiments/README.md` with any new findings.
