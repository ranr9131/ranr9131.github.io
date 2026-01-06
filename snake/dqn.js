class ReplayBuffer {
    constructor(capacity = 10000) {
        this.capacity = capacity;
        this.buffer = [];
        this.position = 0;
    }

    push(state, action, reward, nextState, done) {
        const experience = { state, action, reward, nextState, done };
        if (this.buffer.length < this.capacity) {
            this.buffer.push(experience);
        } else {
            this.buffer[this.position] = experience;
        }
        this.position = (this.position + 1) % this.capacity;
    }

    sample(batchSize) {
        const samples = [];
        const indices = new Set();
        while (indices.size < Math.min(batchSize, this.buffer.length)) {
            indices.add(Math.floor(Math.random() * this.buffer.length));
        }
        for (const idx of indices) {
            samples.push(this.buffer[idx]);
        }
        return samples;
    }

    get length() {
        return this.buffer.length;
    }
}


class DQNAgent {
    constructor(gridSize = 20, config = {}) {
        this.gridSize = gridSize;
        this.nActions = 4;

        // Hyperparameters
        this.gamma = config.gamma || 0.99;
        this.epsilonStart = config.epsilonStart || 1.0;
        this.epsilonEnd = config.epsilonEnd || 0.01;
        this.epsilonDecay = config.epsilonDecay || 0.995;
        this.learningRate = config.learningRate || 0.001;
        this.batchSize = config.batchSize || 64;
        this.targetUpdate = config.targetUpdate || 10;

        this.epsilon = this.epsilonStart;
        this.episode = 0;

        // Replay buffer
        this.replayBuffer = new ReplayBuffer(config.memorySize || 10000);

        // Build models
        this.policyNet = this.buildModel();
        this.targetNet = this.buildModel();
        this.updateTargetNetwork();

        // For feature map visualization
        this.featureMaps = [];
    }

    buildModel() {
        const model = tf.sequential();

        // Conv layers
        model.add(tf.layers.conv2d({
            inputShape: [this.gridSize, this.gridSize, 1],
            filters: 32,
            kernelSize: 3,
            strides: 2,
            activation: 'relu'
        }));

        model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            strides: 2,
            activation: 'relu'
        }));

        model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            strides: 1,
            activation: 'relu'
        }));

        // Flatten and dense layers
        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
        model.add(tf.layers.dense({ units: this.nActions }));

        model.compile({
            optimizer: tf.train.adam(this.learningRate),
            loss: 'meanSquaredError'
        });

        return model;
    }

    updateTargetNetwork() {
        const policyWeights = this.policyNet.getWeights();
        this.targetNet.setWeights(policyWeights);
    }

    stateToTensor(state) {
        // Flatten the 2D state array and reshape to [1, gridSize, gridSize, 1]
        const flat = [];
        for (let y = 0; y < this.gridSize; y++) {
            for (let x = 0; x < this.gridSize; x++) {
                flat.push(state[y][x]);
            }
        }
        return tf.tensor4d(flat, [1, this.gridSize, this.gridSize, 1]);
    }

    async getQValues(state) {
        const stateTensor = this.stateToTensor(state);
        const qValues = await this.policyNet.predict(stateTensor).data();
        stateTensor.dispose();
        return Array.from(qValues);
    }

    async selectAction(state) {
        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * this.nActions);
        } else {
            const qValues = await this.getQValues(state);
            return qValues.indexOf(Math.max(...qValues));
        }
    }

    async train() {
        if (this.replayBuffer.length < this.batchSize) {
            return 0;
        }

        const batch = this.replayBuffer.sample(this.batchSize);

        // Prepare batch data
        const states = batch.map(e => e.state);
        const nextStates = batch.map(e => e.nextState);
        const actions = batch.map(e => e.action);
        const rewards = batch.map(e => e.reward);
        const dones = batch.map(e => e.done ? 1 : 0);

        // Flatten states properly for tensor creation
        const statesFlat = [];
        const nextStatesFlat = [];
        for (let i = 0; i < this.batchSize; i++) {
            for (let y = 0; y < this.gridSize; y++) {
                for (let x = 0; x < this.gridSize; x++) {
                    statesFlat.push(states[i][y][x]);
                    nextStatesFlat.push(nextStates[i][y][x]);
                }
            }
        }

        // Convert to tensors
        const statesTensor = tf.tensor4d(
            statesFlat,
            [this.batchSize, this.gridSize, this.gridSize, 1]
        );
        const nextStatesTensor = tf.tensor4d(
            nextStatesFlat,
            [this.batchSize, this.gridSize, this.gridSize, 1]
        );

        // Get current Q values and target Q values
        const currentQs = this.policyNet.predict(statesTensor);
        const nextQs = this.targetNet.predict(nextStatesTensor);

        // Calculate target values
        const currentQsData = await currentQs.array();
        const nextQsData = await nextQs.array();

        for (let i = 0; i < this.batchSize; i++) {
            const maxNextQ = Math.max(...nextQsData[i]);
            const targetQ = rewards[i] + this.gamma * maxNextQ * (1 - dones[i]);
            currentQsData[i][actions[i]] = targetQ;
        }

        const targetTensor = tf.tensor2d(currentQsData);

        // Train
        const result = await this.policyNet.fit(statesTensor, targetTensor, {
            epochs: 1,
            verbose: 0
        });

        // Cleanup
        statesTensor.dispose();
        nextStatesTensor.dispose();
        currentQs.dispose();
        nextQs.dispose();
        targetTensor.dispose();

        return result.history.loss[0];
    }

    remember(state, action, reward, nextState, done) {
        this.replayBuffer.push(state, action, reward, nextState, done);
    }

    decayEpsilon() {
        this.epsilon = Math.max(this.epsilonEnd, this.epsilon * this.epsilonDecay);
    }

    onEpisodeEnd() {
        this.episode++;
        this.decayEpsilon();

        if (this.episode % this.targetUpdate === 0) {
            this.updateTargetNetwork();
        }
    }

    buildFeatureExtractors() {
        // Build once and reuse
        this.featureExtractors = [];
        for (let layerIdx = 0; layerIdx < 3; layerIdx++) {
            const layer = this.policyNet.layers[layerIdx];
            this.featureExtractors.push(tf.model({
                inputs: this.policyNet.inputs,
                outputs: layer.output
            }));
        }
    }

    async getFeatureMaps(state) {
        if (!this.featureExtractors) {
            this.buildFeatureExtractors();
        }

        const featureMaps = [];
        const stateTensor = this.stateToTensor(state);

        for (let layerIdx = 0; layerIdx < 3; layerIdx++) {
            const activation = this.featureExtractors[layerIdx].predict(stateTensor);
            const data = await activation.array();
            const channels = data[0]; // [height, width, channels]

            const normalized = [];
            const numChannels = Math.min(8, channels[0][0].length);

            for (let c = 0; c < numChannels; c++) {
                let min = Infinity, max = -Infinity;
                const channelData = [];

                for (let y = 0; y < channels.length; y++) {
                    channelData[y] = [];
                    for (let x = 0; x < channels[0].length; x++) {
                        const val = channels[y][x][c];
                        channelData[y][x] = val;
                        min = Math.min(min, val);
                        max = Math.max(max, val);
                    }
                }

                // Normalize
                for (let y = 0; y < channels.length; y++) {
                    for (let x = 0; x < channels[0].length; x++) {
                        channelData[y][x] = (channelData[y][x] - min) / (max - min + 1e-8);
                    }
                }
                normalized.push(channelData);
            }

            featureMaps.push(normalized);
            activation.dispose();
        }

        stateTensor.dispose();
        return featureMaps;
    }
}
