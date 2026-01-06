class SnakeEnv {
    constructor(gridSize = 20) {
        this.gridSize = gridSize;
        this.actionToDirection = {
            0: [-1, 0],  // left
            1: [1, 0],   // right
            2: [0, -1],  // up
            3: [0, 1]    // down
        };
        this.reset();
    }

    reset() {
        const mid = Math.floor(this.gridSize / 2);
        this.snake = [[mid, mid]];
        this.direction = [1, 0];  // Start moving right
        this.placeFood();
        this.score = 0;
        this.steps = 0;
        this.stepsSinceFood = 0;
        this.hungerLimit = 100;
        this.done = false;
        return this.getState();
    }

    placeFood() {
        while (true) {
            this.food = [
                Math.floor(Math.random() * this.gridSize),
                Math.floor(Math.random() * this.gridSize)
            ];
            // Check food is not on snake
            let onSnake = false;
            for (const segment of this.snake) {
                if (segment[0] === this.food[0] && segment[1] === this.food[1]) {
                    onSnake = true;
                    break;
                }
            }
            if (!onSnake) break;
        }
    }

    getState() {
        // Create grid: 0=empty, 1=snake body, 2=snake head, 3=apple
        const grid = [];
        for (let y = 0; y < this.gridSize; y++) {
            grid[y] = new Array(this.gridSize).fill(0);
        }

        // Place snake body
        for (let i = 1; i < this.snake.length; i++) {
            const [x, y] = this.snake[i];
            grid[y][x] = 1;
        }

        // Place snake head
        if (this.snake.length > 0) {
            const [hx, hy] = this.snake[0];
            grid[hy][hx] = 2;
        }

        // Place food
        const [fx, fy] = this.food;
        grid[fy][fx] = 3;

        return grid;
    }

    distanceToFood(pos) {
        return Math.abs(pos[0] - this.food[0]) + Math.abs(pos[1] - this.food[1]);
    }

    step(action) {
        this.steps++;
        this.stepsSinceFood++;

        const newDirection = this.actionToDirection[action];

        // Die on 180-degree turn
        if (newDirection[0] === -this.direction[0] &&
            newDirection[1] === -this.direction[1]) {
            this.done = true;
            return { state: this.getState(), reward: -10, done: true };
        }

        this.direction = newDirection;

        // Calculate new head position
        const head = this.snake[0];
        const newHead = [
            head[0] + this.direction[0],
            head[1] + this.direction[1]
        ];

        // Check wall collision
        if (newHead[0] < 0 || newHead[0] >= this.gridSize ||
            newHead[1] < 0 || newHead[1] >= this.gridSize) {
            this.done = true;
            return { state: this.getState(), reward: -10, done: true };
        }

        // Check self collision
        for (const segment of this.snake) {
            if (segment[0] === newHead[0] && segment[1] === newHead[1]) {
                this.done = true;
                return { state: this.getState(), reward: -10, done: true };
            }
        }

        // Check hunger
        if (this.stepsSinceFood >= this.hungerLimit) {
            this.done = true;
            return { state: this.getState(), reward: -10, done: true };
        }

        // Calculate distance reward
        const oldDistance = this.distanceToFood(head);
        const newDistance = this.distanceToFood(newHead);

        // Move snake
        this.snake.unshift(newHead);

        let reward;
        // Check if food eaten
        if (newHead[0] === this.food[0] && newHead[1] === this.food[1]) {
            this.score++;
            this.placeFood();
            this.stepsSinceFood = 0;
            reward = 20;
        } else {
            this.snake.pop();
            reward = newDistance < oldDistance ? 1 : -1;
        }

        return { state: this.getState(), reward, done: false };
    }
}

// Export for use in other files
if (typeof module !== 'undefined') {
    module.exports = SnakeEnv;
}
