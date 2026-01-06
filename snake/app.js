const GRID_SIZE = 20;
const CELL_SIZE = 20;

let env;
let agent;
let currentEpisode = 0;
let bestScore = 0;

let gameCtx;
let featureCanvases = [];

async function init() {
    console.log('Initializing...');

    try {
        // Setup game canvas
        const gameCanvas = document.getElementById('gameCanvas');
        gameCtx = gameCanvas.getContext('2d');

        // Setup feature map canvases
        const convContainers = [
            document.getElementById('conv1'),
            document.getElementById('conv2'),
            document.getElementById('conv3')
        ];

        for (let layer = 0; layer < 3; layer++) {
            featureCanvases[layer] = [];
            for (let i = 0; i < 8; i++) {
                const canvas = document.createElement('canvas');
                canvas.width = 30;
                canvas.height = 30;
                canvas.className = 'feature-map';
                convContainers[layer].appendChild(canvas);
                featureCanvases[layer].push(canvas);
            }
        }

        // Wait for TensorFlow to be ready
        console.log('Loading TensorFlow.js...');
        await tf.ready();
        console.log('TensorFlow.js ready, backend:', tf.getBackend());

        // Initialize environment and agent
        console.log('Creating environment and agent...');
        env = new SnakeEnv(GRID_SIZE);
        agent = new DQNAgent(GRID_SIZE);
        console.log('Agent created, starting training...');

        // Start training
        trainLoop();
    } catch (error) {
        console.error('Init error:', error);
    }
}

async function trainLoop() {
    try {
        while (currentEpisode < 10000) {
            currentEpisode++;
            let state = env.reset();

            while (!env.done) {
                const qValues = await agent.getQValues(state);
                const action = await agent.selectAction(state);
                const { state: nextState, reward, done } = env.step(action);

                agent.remember(state, action, reward, nextState, done);
                await agent.train();

                // Render
                drawGame(env.getState());
                updateQValues(qValues, action);
                updateStats();

                // Feature maps
                try {
                    const featureMaps = await agent.getFeatureMaps(state);
                    drawFeatureMaps(featureMaps);
                } catch (e) {}

                state = nextState;
                await sleep(20);
            }

            agent.onEpisodeEnd();
            if (env.score > bestScore) {
                bestScore = env.score;
            }
            updateStats();
        }
    } catch (error) {
        console.error('Training error:', error);
    }
}

function drawGame(grid) {
    gameCtx.fillStyle = '#000';
    gameCtx.fillRect(0, 0, GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE);

    for (let y = 0; y < GRID_SIZE; y++) {
        for (let x = 0; x < GRID_SIZE; x++) {
            const cell = grid[y][x];
            if (cell === 1) {
                gameCtx.fillStyle = '#0a0';
            } else if (cell === 2) {
                gameCtx.fillStyle = '#0f0';
            } else if (cell === 3) {
                gameCtx.fillStyle = '#f00';
            } else {
                continue;
            }
            gameCtx.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1);
        }
    }
}

function updateQValues(qValues, action) {
    const ids = ['q-left', 'q-right', 'q-up', 'q-down'];
    const valIds = ['q-left-val', 'q-right-val', 'q-up-val', 'q-down-val'];

    ids.forEach((id, i) => {
        const el = document.getElementById(id);
        const valEl = document.getElementById(valIds[i]);
        el.classList.toggle('selected', i === action);
        valEl.textContent = qValues[i].toFixed(2);
    });
}

function drawFeatureMaps(featureMaps) {
    featureMaps.forEach((layer, layerIdx) => {
        if (layerIdx >= 3) return;
        layer.forEach((fmap, mapIdx) => {
            if (mapIdx >= 8) return;
            const canvas = featureCanvases[layerIdx][mapIdx];
            const ctx = canvas.getContext('2d');
            const h = fmap.length;
            const w = fmap[0].length;
            const scaleX = canvas.width / w;
            const scaleY = canvas.height / h;

            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    const val = Math.floor(fmap[y][x] * 255);
                    ctx.fillStyle = `rgb(${val},${val},${val})`;
                    ctx.fillRect(x * scaleX, y * scaleY, Math.ceil(scaleX), Math.ceil(scaleY));
                }
            }
        });
    });
}

function updateStats() {
    document.getElementById('episode').textContent = currentEpisode;
    document.getElementById('score').textContent = env ? env.score : 0;
    document.getElementById('epsilon').textContent = agent ? agent.epsilon.toFixed(3) : '1.000';
    document.getElementById('bestScore').textContent = bestScore;
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Start when page loads
window.addEventListener('load', init);
