const trainData = {
  sizeMB: [
    0.08,
    9.0,
    0.001,
    0.1,
    8.0,
    5.0,
    0.1,
    6.0,
    0.05,
    0.5,
    0.002,
    2.0,
    0.005,
    10.0,
    0.01,
    7.0,
    6.0,
    5.0,
    1.0,
    1.0,
  ],
  timeSec: [
    0.135,
    0.739,
    0.067,
    0.126,
    0.646,
    0.435,
    0.069,
    0.497,
    0.068,
    0.116,
    0.07,
    0.289,
    0.076,
    0.744,
    0.083,
    0.56,
    0.48,
    0.399,
    0.153,
    0.149,
  ],
};
const testData = {
  sizeMB: [
    5.0,
    0.2,
    0.001,
    9.0,
    0.002,
    0.02,
    0.008,
    4.0,
    0.001,
    1.0,
    0.005,
    0.08,
    0.8,
    0.2,
    0.05,
    7.0,
    0.005,
    0.002,
    8.0,
    0.008,
  ],
  timeSec: [
    0.425,
    0.098,
    0.052,
    0.686,
    0.066,
    0.078,
    0.07,
    0.375,
    0.058,
    0.136,
    0.052,
    0.063,
    0.183,
    0.087,
    0.066,
    0.558,
    0.066,
    0.068,
    0.61,
    0.057,
  ],
};
// Convert data to Tensors.
const trainTensors = {
  sizeMB: tf.tensor2d(trainData.sizeMB, [20, 1]),
  timeSec: tf.tensor2d(trainData.timeSec, [20, 1]),
};
const testTensors = {
  sizeMB: tf.tensor2d(testData.sizeMB, [20, 1]),
  timeSec: tf.tensor2d(testData.timeSec, [20, 1]),
};

////
// Data Markers
////
const dataTraceTrain = {
  x: trainData.sizeMB,
  y: trainData.timeSec,
  name: "trainData",
  mode: "markers",
  type: "scatter",
  marker: { symbol: "circle", size: 8 },
};
const dataTraceTest = {
  x: testData.sizeMB,
  y: testData.timeSec,
  name: "testData",
  mode: "markers",
  type: "scatter",
  marker: { symbol: "triangle-up", size: 10 },
};
const dataTrace10Epochs = {
  x: [0, 2],
  y: [0, 0.01],
  name: "model after N epochs",
  mode: "lines",
  line: { color: "blue", width: 1, dash: "dot" },
};
const dataTrace20Epochs = {
  x: [0, 2],
  y: [0, 0.01],
  name: "model after N epochs",
  mode: "lines",
  line: { color: "green", width: 2, dash: "dash" },
};
const dataTrace100Epochs = {
  x: [0, 2],
  y: [0, 0.01],
  name: "model after N epochs",
  mode: "lines",
  line: { color: "red", width: 3, dash: "longdash" },
};
const dataTrace200Epochs = {
  x: [0, 2],
  y: [0, 0.01],
  name: "model after N epochs",
  mode: "lines",
  line: { color: "black", width: 4, dash: "solid" },
};

////
// Set up plotly plot.
////
Plotly.newPlot(
  "dataSpaceWith4Lines",
  [
    dataTraceTrain,
    dataTraceTest,
    dataTrace10Epochs,
    dataTrace20Epochs,
    dataTrace100Epochs,
    dataTrace200Epochs,
  ],
  {
    width: 700,
    title: "Model fit result",
    xaxis: {
      title: "size (MB)",
    },
    yaxis: {
      title: "time (sec)",
    },
  }
);

////
// Construct and compile model.
////
const model = tf.sequential();
model.add(
  tf.layers.dense({
    units: 1,
    inputShape: [1],
  })
);
// Use a slower learning rate for illustration purposes.
const optimizer = tf.train.sgd(0.0005);
model.compile({ optimizer: optimizer, loss: "meanAbsoluteError" });

// Updates a specified line on the plot
function updateScatterWithLines(dataTrace, k, b, N, traceIndex) {
  dataTrace.x = [0, 10];
  dataTrace.y = [b, b + k * 10];
  var update = {
    x: [dataTrace.x],
    y: [dataTrace.y],
    name: "model after " + N + " epochs",
  };
  Plotly.restyle("dataSpaceWith4Lines", update, traceIndex);
}

// Initialize to k=0, b=0 for pretty illustration purposes.
// You may want to remove this to see how the model looks with
// random initialization
let k = 0;
let b = 0;
model.setWeights([tf.tensor2d([k], [1, 1]), tf.tensor1d([b])]);

////
// Train the model within an async block
////
(async () => {
  await model.fit(trainTensors.sizeMB, trainTensors.timeSec, {
    epochs: 200,
    // Use callbacks to orchestrate certain functions at certain triggers.
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        k = model.getWeights()[0].dataSync()[0];
        b = model.getWeights()[1].dataSync()[0];
        // console.log(`epoch ${epoch}`);
        if (epoch === 9) {
          updateScatterWithLines(dataTrace10Epochs, k, b, 10, 2);
          console.log("wrote model 10");
        }
        if (epoch === 19) {
          updateScatterWithLines(dataTrace20Epochs, k, b, 20, 3);
          console.log("wrote model 20");
        }
        if (epoch === 99) {
          updateScatterWithLines(dataTrace100Epochs, k, b, 100, 4);
          console.log("wrote model 100");
        }
        if (epoch === 199) {
          updateScatterWithLines(dataTrace200Epochs, k, b, 200, 5);
          console.log("wrote model 200");
        }
        await tf.nextFrame();
      },
    },
  });
})();
