# 🎮 Tetris Game

A classic Tetris game implementation using vanilla JavaScript, HTML5 Canvas, and CSS. Built with modern web technologies, featuring a clean interface and smooth gameplay.

![Tetris Game](https://img.shields.io/badge/Game-Tetris-brightgreen)
![JavaScript](https://img.shields.io/badge/Language-JavaScript-yellow)
![License](https://img.shields.io/badge/License-MIT-blue)

## 🚀 Features

- **Classic Gameplay**: All 7 Tetromino pieces with authentic Tetris mechanics
- **Responsive Controls**: Smooth keyboard controls for piece movement and rotation
- **Scoring System**: Points awarded for line clears with level progression
- **Visual Feedback**: Next piece preview and real-time stats display
- **Modern Design**: Clean, dark-themed UI with vibrant piece colors
- **Progressive Difficulty**: Game speed increases with level advancement

## 🎯 How to Play

### Controls
- **←/→** - Move piece left/right
- **↓** - Soft drop (faster fall)
- **↑** - Rotate piece clockwise
- **Space** - Hard drop (instant drop)
- **Start Game** - Begin a new game
- **Pause** - Pause/resume current game

### Gameplay
1. Arrange falling Tetromino pieces to create complete horizontal lines
2. Complete lines are cleared and award points
3. Game speed increases every 10 lines cleared
4. Game ends when pieces reach the top of the board

## 🛠️ Installation

### Option 1: Play Locally
```bash
# Clone the repository
git clone https://github.com/bananacan1/tetris-game.git

# Navigate to the game directory
cd tetris-game

# Open in browser (macOS)
open tetris.html

# Or open in browser (Windows/Linux)
xdg-open tetris.html  # Linux
start tetris.html     # Windows
```

### Option 2: Play Online
Visit the live demo: [Play Tetris](https://bananacan1.github.io/tetris-game/tetris.html) *(GitHub Pages deployment)*

## 📁 Project Structure

```
tetris-game/
├── tetris.html      # Main HTML file
├── style.css        # Game styling
├── tetris.js        # Game logic
├── README.md        # Documentation
└── LICENSE          # MIT License
```

## 🎨 Technical Details

### Technologies Used
- **HTML5 Canvas** for rendering the game board and pieces
- **Vanilla JavaScript** for game logic (no frameworks)
- **CSS3** for styling and responsive design
- **RequestAnimationFrame** for smooth game loop

### Key Features Implementation
- **Collision Detection**: Boundary and piece collision checking
- **Piece Rotation**: Matrix transformation for clockwise rotation
- **Line Clearing**: Row completion detection and removal
- **Score Calculation**: Points based on lines cleared and level

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contributions
- Add sound effects and background music
- Implement hold piece functionality
- Add ghost piece (drop preview)
- Create different game modes
- Add touch controls for mobile
- Implement high score persistence

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the original Tetris game by Alexey Pajitnov
- Built as a learning project for HTML5 Canvas and game development
- Thanks to all contributors and players!

## 📊 Browser Compatibility

- Chrome (recommended)
- Firefox
- Safari
- Edge
- Opera

*Note: Requires a modern browser with ES6 support*

---

Made with ❤️ by [bananacan1](https://github.com/bananacan1)