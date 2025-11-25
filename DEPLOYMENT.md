# 🚀 Deploying SemanticJury to Hugging Face Spaces

This guide will walk you through deploying your semantic legal search engine to Hugging Face Spaces.

## Prerequisites

1. **Hugging Face Account**: Sign up at https://huggingface.co/join
2. **Git**: Installed and configured on your machine
3. **Prepared Data**: Run `prepare_data.py` locally first to create the database

## Step-by-Step Deployment

### 1. Prepare Your Local Repository

First, make sure everything is working locally:

```bash
cd /Users/shreyamendi/SemanticJury

# Activate virtual environment
source venv/bin/activate

# Prepare the data (this creates chromadb/ and citation_graph.json)
python3 prepare_data.py

# Test the app locally
python3 app.py
```

### 2. Create a Hugging Face Space

1. Go to https://huggingface.co/new-space
2. Fill in the details:
   - **Owner**: Your username or organization
   - **Space name**: `SemanticJury` (or your preferred name)
   - **License**: Choose appropriate license (e.g., MIT)
   - **Select the Space SDK**: Choose **Gradio**
   - **Space hardware**: Start with **CPU basic** (free tier)
   - **Visibility**: Public or Private

3. Click **Create Space**

### 3. Initialize Git Repository (if not already done)

```bash
cd /Users/shreyamendi/SemanticJury

# Initialize git if needed
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit: Semantic legal search with citations"
```

### 4. Add Hugging Face Remote and Push

```bash
# Add Hugging Face as a remote
# Replace YOUR_USERNAME with your Hugging Face username
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/SemanticJury

# Push to Hugging Face
git push hf main
```

If you get an authentication error:
- Go to https://huggingface.co/settings/tokens
- Create a new token with "write" access
- Use the token as your password when prompted

### 5. Important Files for Deployment

Your repository should include these files:

```
SemanticJury/
├── app.py                    # Main application (REQUIRED)
├── prepare_data.py           # Data preparation script
├── visualize.py              # Visualization utilities
├── requirements.txt          # Dependencies (REQUIRED)
├── README.md                 # Documentation
├── USAGE_GUIDE.md           # Usage instructions
├── chromadb/                 # Vector database (REQUIRED)
│   └── chroma.sqlite3
└── citation_graph.json       # Citation data (REQUIRED)
```

### 6. Space Configuration (Optional)

Create a file called `README.md` in your repo root (if you want to customize the Space page):

```markdown
---
title: SemanticJury
emoji: ⚖️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
license: mit
---

# SemanticJury: Legal Search with Citation Support

Semantic search engine for legal documents with citation tracking and provenance.

[Add your description here]
```

### 7. Monitor Deployment

1. Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/SemanticJury`
2. Watch the build logs in the "Logs" tab
3. Wait for the build to complete (usually 3-5 minutes)
4. Once ready, your app will be accessible at the Space URL

### 8. Troubleshooting Common Issues

#### Issue: "ChromaDB not found"
**Solution**: Make sure you committed the `chromadb/` directory:
```bash
git add chromadb/
git commit -m "Add ChromaDB database"
git push hf main
```

#### Issue: "Out of memory"
**Solution**: The free tier has limited RAM. Consider:
- Using a smaller embedding model
- Reducing the dataset size
- Upgrading to a paid Space tier

#### Issue: "Build timeout"
**Solution**: Large dependencies can timeout. Try:
- Remove unused dependencies from requirements.txt
- Use CPU-only torch: `torch>=2.0.0` instead of full version

#### Issue: "Port already in use"
**Solution**: Hugging Face Spaces automatically handles ports. Make sure your `app.py` ends with:
```python
if __name__ == "__main__":
    demo.launch()
```
Not:
```python
demo.launch(server_port=7860)  # Don't specify port
```

### 9. Updating Your Space

To update your deployed Space:

```bash
# Make your changes
git add .
git commit -m "Description of changes"
git push hf main
```

The Space will automatically rebuild with your changes.

### 10. Alternative: Using Hugging Face CLI

You can also use the Hugging Face CLI:

```bash
# Install the CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Upload files
huggingface-cli upload YOUR_USERNAME/SemanticJury . --repo-type space
```

## Advanced Configuration

### Custom Domain

If you have a Pro subscription, you can set up a custom domain:
1. Go to Space settings
2. Navigate to "Custom domain"
3. Follow the instructions

### Persistent Storage

For larger datasets:
1. Go to Space settings
2. Enable "Persistent storage"
3. Store your ChromaDB in the persistent volume

### Hardware Upgrade

For better performance:
1. Go to Space settings
2. Click "Change hardware"
3. Select a more powerful option (requires payment)

## Environment Variables

If you need API keys or secrets:

1. Go to Space settings
2. Navigate to "Variables and secrets"
3. Add your secrets (they won't be visible in logs)

In your code:
```python
import os
api_key = os.environ.get("MY_API_KEY")
```

## Best Practices

1. **Test Locally First**: Always test your app locally before deploying
2. **Small Database**: Keep your ChromaDB small for faster builds
3. **Clear Documentation**: Add clear README with usage instructions
4. **Error Handling**: Add try/except blocks for robustness
5. **Loading States**: Show loading messages for better UX
6. **Version Control**: Use git tags for releases

## Monitoring and Analytics

Hugging Face Spaces provides:
- **Usage stats**: Number of visitors and sessions
- **Build logs**: Deployment and runtime logs
- **Hardware usage**: RAM and CPU usage

Access these in your Space settings.

## Community Features

1. **Discussions**: Enable discussions for user feedback
2. **Duplicates**: Allow users to duplicate your Space
3. **Embedding**: Embed your Space in other websites
4. **API Access**: Spaces have automatic API endpoints

## Cost Considerations

- **Free Tier**:
  - 2 CPU cores
  - 16 GB RAM
  - Public Spaces only
  - Sleep after inactivity

- **Paid Tiers**: Starting at $5/month
  - More resources
  - Always-on option
  - Private Spaces
  - GPU access

## Example Deployment Commands

Complete workflow:

```bash
# 1. Prepare data locally
cd /Users/shreyamendi/SemanticJury
source venv/bin/activate
python3 prepare_data.py

# 2. Test locally
python3 app.py

# 3. Commit everything
git add .
git commit -m "Ready for deployment"

# 4. Push to Hugging Face
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/SemanticJury
git push hf main

# 5. Visit your Space
# https://huggingface.co/spaces/YOUR_USERNAME/SemanticJury
```

## Support

- **Documentation**: https://huggingface.co/docs/hub/spaces
- **Forum**: https://discuss.huggingface.co/
- **Discord**: https://hf.co/join/discord

## Success Checklist

Before deploying, ensure:

- [ ] `prepare_data.py` has been run
- [ ] `chromadb/` directory exists and is committed
- [ ] `citation_graph.json` exists and is committed
- [ ] `requirements.txt` includes all dependencies
- [ ] `app.py` works locally
- [ ] README.md is informative
- [ ] Git repository is initialized
- [ ] Hugging Face account is created
- [ ] Space is created on Hugging Face

Once deployed, verify:

- [ ] Space builds successfully
- [ ] App loads without errors
- [ ] Search functionality works
- [ ] Citation explorer works
- [ ] Visualizations generate correctly
- [ ] No missing dependencies

## Next Steps After Deployment

1. Share your Space on social media
2. Add it to your portfolio
3. Write a blog post about it
4. Submit to Hugging Face showcase
5. Gather user feedback
6. Iterate and improve

Good luck with your deployment! 🚀
