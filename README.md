# McFly Enhanced - Smarter Shell History Search

<img src="/docs/screenshot.png" alt="screenshot" width="400">

## 🚀 **Personal Enhanced Fork**

This is a personal enhanced fork of McFly with **improved command suggestions** and **better search experience**.

**What's Better in This Fork:**
- **🧠 Smarter Suggestions**: Advanced neural network learns your patterns better
- **⚡ Better Search Experience**: Fuzzy search with **bold highlighting** makes matches easy to spot
- **🎯 More Accurate Results**: Improved algorithm gives you the right command faster
- **📊 Better Context Understanding**: Understands your workflow patterns more intelligently
- **⌨️ Better Terminal Support**: Updated keybindings (F4 instead of F1) for wider compatibility

---

## 🎯 **What McFly Enhanced Does**

McFly Enhanced replaces your `ctrl-r` shell history search with an **intelligent search** that actually understands what you're looking for. Instead of just matching text, it learns your habits and suggests commands that make sense for your current situation.

**Why You'll Love It:**
- 🔍 **Smart History Search**: Find commands faster with context-aware suggestions
- � **Location Awareness**: Suggests commands you typically run in your current directory
- � **Workflow Understanding**: Remembers command sequences and suggests what usually comes next
- ✨ **Easy to Spot Matches**: Bold highlighting makes it obvious what matched your search
- 🗑️ **Clean History**: Easy removal of commands you don't want to see again
- 🌐 **Works Everywhere**: Supports Bash, Zsh, Fish, and PowerShell

---

## 🎯 **How It's Smarter**

### **Better Command Suggestions**
The enhanced version learns from your behavior and suggests commands that actually make sense:

- 📁 **Directory-Smart**: Remembers which commands you use in which folders
- 🔗 **Sequence-Aware**: Learns common command workflows (like `git add` → `git commit`)
- 📊 **Usage-Based**: Prioritizes commands you actually use frequently
- ⏰ **Time-Aware**: Recent commands get appropriate priority
- ✅ **Success-Focused**: Avoids suggesting commands that usually fail
- 🎯 **Learning**: Gets better the more you use it

### **Enhanced Search Experience**
- **Bold Match Highlighting**: Easy to see exactly what matched your search
- **Fuzzy Search**: Find commands even with typos or partial matches
- **Faster Results**: Improved algorithm delivers suggestions quicker

---

## 📦 **Installation**

### **Install This Enhanced Version**

```bash
# Clone this enhanced fork
git clone https://github.com/wstein/mcfly.git
cd mcfly

# Build and install
cargo install --path .
```

### **Shell Setup**

Add to your shell configuration file:

**Bash** (`~/.bashrc`):
```bash
eval "$(mcfly init bash)"
```

**Zsh** (`~/.zshrc`):
```bash
eval "$(mcfly init zsh)"
```

**Fish** (`~/.config/fish/config.fish`):
```bash
mcfly init fish | source
```

**PowerShell** (`$PROFILE`):
```powershell
Invoke-Expression -Command $(mcfly init powershell | out-string)
```

Then restart your terminal or run `. ~/.bashrc` / `. ~/.zshrc` / `source ~/.config/fish/config.fish`

---

## ⚙️ **Recommended Settings**

### **Enable Enhanced Fuzzy Search**
```bash
# Add to your shell config file
export MCFLY_FUZZY=3
```

### **Optimize Results**
```bash
# Show more results (default: 30)
export MCFLY_RESULTS=50

# Sort by relevance (recommended for smart suggestions)
export MCFLY_RESULTS_SORT=RANK
```

For all configuration options, see the [original McFly documentation](https://github.com/cantino/mcfly#settings).

---

## � **Usage Tips**

### **Getting the Best Results**
- **Use it regularly**: The smart suggestions get better as you use McFly more
- **Enable fuzzy search**: Set `MCFLY_FUZZY=3` for the best search experience  
- **Try partial matches**: You don't need to remember exact command names
- **Use directory context**: McFly learns which commands you use in which folders

### **Common Commands**
- `ctrl-r`: Start McFly search
- `↑/↓`: Navigate suggestions
- `Enter`: Run selected command
- `Tab`: Select command but don't run (edit first)
- `Delete`: Remove unwanted commands from history
- `F4`: Toggle sort order (improved from F1 for better terminal support)

---

## 🔍 **What Makes This Fork Different**

| Feature | Original McFly | This Enhanced Fork |
|---------|---------------|-------------------|
| **Suggestions Quality** | Good | Much smarter with better learning |
| **Search Highlighting** | Basic | **Bold highlighting** - easy to spot matches |
| **Learning Speed** | Standard | Faster adaptation to your patterns |
| **Command Context** | Basic | Better understanding of workflows |
| **Setup** | Standard | Same easy setup, better results |

---

## 🆘 **Troubleshooting & More Info**

This enhanced fork maintains full compatibility with original McFly:

- **All original features work**: Same interface, same shell support, with improved keybindings (F4 vs F1)
- **Existing history preserved**: Your command history transfers automatically  
- **Same configuration**: All original environment variables work

For detailed troubleshooting, shell-specific setup help, and advanced features, see the comprehensive [original McFly documentation](https://github.com/cantino/mcfly).

---

## � **Learn More**

- **Original McFly Project**: [cantino/mcfly](https://github.com/cantino/mcfly) - Full documentation and community
- **Shell Integration Help**: See original docs for detailed shell-specific setup instructions
- **All Settings**: Complete list of environment variables in original documentation
- **Issues & Support**: Use the original project's resources for general McFly questions

---

## 🙏 **Credits**

This personal enhanced fork builds upon the excellent [original McFly project by cantino](https://github.com/cantino/mcfly). All core functionality remains faithful to the original design while adding meaningful improvements to the search experience.
