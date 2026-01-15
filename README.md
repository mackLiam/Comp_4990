# Comp_4990

Computer vision system for object detection, tracking, and 3D reconstruction.

## Team Members
- Haydar Beydoun
- Liam Mackenzie  
- Jabari Namuro

## Setup Instructions (Git Bash)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Create virtual environment
```bash
py -m venv venv
source venv/Scripts/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Test the installation
```bash
python main.py
```

### 5. Add test videos
Place your test videos in `data/input_videos/`

## Project Structure
- `src/` - Source code
- `data/` - Input videos and outputs
- `tests/` - Test scripts
- `docs/` - Documentation

## Current Status
- [x] Environment setup
- [ ] Video input handler
- [ ] Object detection
- [ ] Object tracking
- [ ] 3D reconstruction (optional)

## Deactivating Virtual Environment
When you're done working:
```bash
deactivate
```
