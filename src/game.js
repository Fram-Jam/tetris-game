/**
 * Tetris Game Implementation
 * A classic Tetris game built with HTML5 Canvas and vanilla JavaScript
 */

// Canvas elements for game board and next piece preview
const canvas = document.getElementById('game-board');
const ctx = canvas.getContext('2d');
const nextCanvas = document.getElementById('next-piece-canvas');
const nextCtx = nextCanvas.getContext('2d');

// Game constants
const BLOCK_SIZE = 30;  // Size of each block in pixels
const BOARD_WIDTH = 10;  // Game board width in blocks
const BOARD_HEIGHT = 20; // Game board height in blocks

// Color palette for pieces (0 = empty, 1-7 = different pieces)
const COLORS = {
    0: '#000000',  // Empty
    1: '#FF0D72',  // I-piece (pink)
    2: '#0DC2FF',  // O-piece (cyan)
    3: '#0DFF72',  // T-piece (green)
    4: '#F538FF',  // L-piece (purple)
    5: '#FF8E0D',  // J-piece (orange)
    6: '#FFE138',  // S-piece (yellow)
    7: '#3877FF'   // Z-piece (blue)
};

// Tetromino piece definitions
// Each piece is represented as a 2D array where 1 = filled, 0 = empty
const PIECES = [
    [[1, 1, 1, 1]],              // I-piece
    [[1, 1], [1, 1]],            // O-piece
    [[1, 1, 1], [0, 1, 0]],      // T-piece
    [[1, 1, 1], [1, 0, 0]],      // L-piece
    [[1, 1, 1], [0, 0, 1]],      // J-piece
    [[0, 1, 1], [1, 1, 0]],      // S-piece
    [[1, 1, 0], [0, 1, 1]]       // Z-piece
];

/**
 * Main Tetris game class
 * Handles game state, piece movement, collision detection, and rendering
 */
class Tetris {
    constructor() {
        // Initialize game board as 2D array (20x10)
        this.board = Array(BOARD_HEIGHT).fill(null).map(() => Array(BOARD_WIDTH).fill(0));
        
        // Current piece state
        this.currentPiece = null;
        this.currentX = 0;      // Piece X position
        this.currentY = 0;      // Piece Y position
        this.currentColor = 0;  // Piece color index
        
        // Next piece preview
        this.nextPiece = null;
        this.nextColor = 0;
        
        // Game statistics
        this.score = 0;
        this.level = 1;
        this.lines = 0;
        
        // Game state flags
        this.gameOver = false;
        this.paused = false;
        
        // Drop timing
        this.dropCounter = 0;      // Time accumulator
        this.dropInterval = 1000;  // Drop speed in ms
        this.lastTime = 0;         // Last frame timestamp
        
        this.init();
    }
    
    init() {
        this.spawnPiece();
        this.generateNextPiece();
        this.updateStats();
    }
    
    generateNextPiece() {
        const pieceIndex = Math.floor(Math.random() * PIECES.length);
        this.nextPiece = PIECES[pieceIndex];
        this.nextColor = pieceIndex + 1;
        this.drawNextPiece();
    }
    
    spawnPiece() {
        if (this.nextPiece) {
            this.currentPiece = this.nextPiece;
            this.currentColor = this.nextColor;
            this.generateNextPiece();
        } else {
            const pieceIndex = Math.floor(Math.random() * PIECES.length);
            this.currentPiece = PIECES[pieceIndex];
            this.currentColor = pieceIndex + 1;
        }
        
        this.currentX = Math.floor((BOARD_WIDTH - this.currentPiece[0].length) / 2);
        this.currentY = 0;
        
        if (this.collision()) {
            this.gameOver = true;
        }
    }
    
    collision() {
        for (let y = 0; y < this.currentPiece.length; y++) {
            for (let x = 0; x < this.currentPiece[y].length; x++) {
                if (this.currentPiece[y][x]) {
                    const boardX = this.currentX + x;
                    const boardY = this.currentY + y;
                    
                    if (boardX < 0 || boardX >= BOARD_WIDTH || 
                        boardY >= BOARD_HEIGHT ||
                        (boardY >= 0 && this.board[boardY][boardX])) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
    
    merge() {
        for (let y = 0; y < this.currentPiece.length; y++) {
            for (let x = 0; x < this.currentPiece[y].length; x++) {
                if (this.currentPiece[y][x]) {
                    const boardY = this.currentY + y;
                    if (boardY >= 0) {
                        this.board[boardY][this.currentX + x] = this.currentColor;
                    }
                }
            }
        }
    }
    
    rotate() {
        const rotated = this.currentPiece[0].map((_, i) => 
            this.currentPiece.map(row => row[i]).reverse()
        );
        
        const previousPiece = this.currentPiece;
        this.currentPiece = rotated;
        
        if (this.collision()) {
            this.currentPiece = previousPiece;
        }
    }
    
    moveLeft() {
        this.currentX--;
        if (this.collision()) {
            this.currentX++;
        }
    }
    
    moveRight() {
        this.currentX++;
        if (this.collision()) {
            this.currentX--;
        }
    }
    
    moveDown() {
        this.currentY++;
        if (this.collision()) {
            this.currentY--;
            this.merge();
            this.clearLines();
            this.spawnPiece();
        }
        this.dropCounter = 0;
    }
    
    hardDrop() {
        while (!this.collision()) {
            this.currentY++;
        }
        this.currentY--;
        this.merge();
        this.clearLines();
        this.spawnPiece();
    }
    
    clearLines() {
        let linesCleared = 0;
        
        for (let y = BOARD_HEIGHT - 1; y >= 0; y--) {
            if (this.board[y].every(cell => cell !== 0)) {
                this.board.splice(y, 1);
                this.board.unshift(Array(BOARD_WIDTH).fill(0));
                linesCleared++;
                y++;
            }
        }
        
        if (linesCleared > 0) {
            this.lines += linesCleared;
            this.score += linesCleared * 100 * this.level;
            this.level = Math.floor(this.lines / 10) + 1;
            this.dropInterval = Math.max(100, 1000 - (this.level - 1) * 100);
            this.updateStats();
        }
    }
    
    updateStats() {
        document.getElementById('score').textContent = this.score;
        document.getElementById('level').textContent = this.level;
        document.getElementById('lines').textContent = this.lines;
    }
    
    draw() {
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        for (let y = 0; y < BOARD_HEIGHT; y++) {
            for (let x = 0; x < BOARD_WIDTH; x++) {
                if (this.board[y][x]) {
                    this.drawBlock(ctx, x, y, COLORS[this.board[y][x]]);
                }
            }
        }
        
        if (this.currentPiece) {
            for (let y = 0; y < this.currentPiece.length; y++) {
                for (let x = 0; x < this.currentPiece[y].length; x++) {
                    if (this.currentPiece[y][x]) {
                        this.drawBlock(ctx, this.currentX + x, this.currentY + y, COLORS[this.currentColor]);
                    }
                }
            }
        }
        
        if (this.gameOver) {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#fff';
            ctx.font = '30px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('GAME OVER', canvas.width / 2, canvas.height / 2);
        }
        
        if (this.paused && !this.gameOver) {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#fff';
            ctx.font = '30px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('PAUSED', canvas.width / 2, canvas.height / 2);
        }
    }
    
    drawBlock(context, x, y, color) {
        context.fillStyle = color;
        context.fillRect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE - 1, BLOCK_SIZE - 1);
        
        context.fillStyle = 'rgba(255, 255, 255, 0.2)';
        context.fillRect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE - 1, 2);
        context.fillRect(x * BLOCK_SIZE, y * BLOCK_SIZE, 2, BLOCK_SIZE - 1);
        
        context.fillStyle = 'rgba(0, 0, 0, 0.2)';
        context.fillRect(x * BLOCK_SIZE + BLOCK_SIZE - 3, y * BLOCK_SIZE, 2, BLOCK_SIZE - 1);
        context.fillRect(x * BLOCK_SIZE, y * BLOCK_SIZE + BLOCK_SIZE - 3, BLOCK_SIZE - 1, 2);
    }
    
    drawNextPiece() {
        nextCtx.fillStyle = '#000';
        nextCtx.fillRect(0, 0, nextCanvas.width, nextCanvas.height);
        
        if (this.nextPiece) {
            const offsetX = (4 - this.nextPiece[0].length) / 2;
            const offsetY = (4 - this.nextPiece.length) / 2;
            
            for (let y = 0; y < this.nextPiece.length; y++) {
                for (let x = 0; x < this.nextPiece[y].length; x++) {
                    if (this.nextPiece[y][x]) {
                        this.drawBlock(nextCtx, offsetX + x, offsetY + y, COLORS[this.nextColor]);
                    }
                }
            }
        }
    }
    
    update(time = 0) {
        if (!this.gameOver && !this.paused) {
            const deltaTime = time - this.lastTime;
            this.lastTime = time;
            
            this.dropCounter += deltaTime;
            
            if (this.dropCounter > this.dropInterval) {
                this.moveDown();
            }
        }
        
        this.draw();
        requestAnimationFrame(this.update.bind(this));
    }
}

let game = null;

document.getElementById('start-btn').addEventListener('click', () => {
    game = new Tetris();
    game.update();
});

document.getElementById('pause-btn').addEventListener('click', () => {
    if (game) {
        game.paused = !game.paused;
    }
});

document.addEventListener('keydown', (e) => {
    if (!game || game.gameOver || game.paused) return;
    
    switch(e.key) {
        case 'ArrowLeft':
            e.preventDefault();
            game.moveLeft();
            break;
        case 'ArrowRight':
            e.preventDefault();
            game.moveRight();
            break;
        case 'ArrowDown':
            e.preventDefault();
            game.moveDown();
            break;
        case 'ArrowUp':
            e.preventDefault();
            game.rotate();
            break;
        case ' ':
            e.preventDefault();
            game.hardDrop();
            break;
    }
});