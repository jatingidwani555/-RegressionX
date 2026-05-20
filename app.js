// Global State
let dataset = [];
let columns = [];
let targetVar = null;
let independentVars = [];

let modelCoefs = []; // [intercept, beta1, beta2, ...]
let userCoefs = []; // User tweaked coefs
let isMultiple = false;

// DOM Elements
const csvUpload = document.getElementById('csv-upload');
const fileNameDisplay = document.getElementById('file-name-display');
const varSelectionDiv = document.getElementById('variable-selection');
const targetVarSelect = document.getElementById('target-var');
const indepVarsContainer = document.getElementById('independent-vars-container');
const modelControlsDiv = document.getElementById('model-controls');
const slidersContainer = document.getElementById('sliders-container');
const resetBtn = document.getElementById('reset-btn');
const metricsPanel = document.getElementById('metrics-panel');
const r2ScoreDisplay = document.getElementById('r2-score');
const mseScoreDisplay = document.getElementById('mse-score');
const equationDisplay = document.getElementById('equation-display');
const emptyState = document.getElementById('empty-state');
const chartsGrid = document.getElementById('charts-grid');
const toggle3dBtn = document.getElementById('toggle-3d');

// Event Listeners
csvUpload.addEventListener('change', handleFileUpload);
targetVarSelect.addEventListener('change', updateRegression);
resetBtn.addEventListener('click', () => {
    userCoefs = [...modelCoefs];
    updateSliders();
    updatePlots();
});

let is3D = false;
toggle3dBtn.addEventListener('click', () => {
    is3D = !is3D;
    updatePlots();
});

// File Upload & Parsing
function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    fileNameDisplay.textContent = file.name;
    
    Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: function(results) {
            processData(results.data);
        }
    });
}

function processData(data) {
    // Filter numeric columns
    if(data.length === 0) return;
    const firstRow = data[0];
    columns = Object.keys(firstRow).filter(key => typeof firstRow[key] === 'number');
    
    if (columns.length < 2) {
        alert("Dataset must have at least 2 numerical columns.");
        return;
    }

    // Filter out rows with nulls in numerical columns
    dataset = data.filter(row => columns.every(col => row[col] !== null && row[col] !== undefined));

    populateVariableSelectors();
    varSelectionDiv.style.display = 'block';
    emptyState.style.display = 'none';
    chartsGrid.style.display = 'grid';
    metricsPanel.style.display = 'flex';
    
    // Default selection
    targetVar = columns[columns.length - 1];
    targetVarSelect.value = targetVar;
    independentVars = [columns[0]];
    
    updateCheckboxSelection();
    updateRegression();
}

function populateVariableSelectors() {
    targetVarSelect.innerHTML = '';
    indepVarsContainer.innerHTML = '';

    columns.forEach(col => {
        // Target Dropdown
        const option = document.createElement('option');
        option.value = col;
        option.textContent = col;
        targetVarSelect.appendChild(option);

        // Independent Checkboxes
        const label = document.createElement('label');
        label.className = 'checkbox-item';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.value = col;
        checkbox.addEventListener('change', handleIndepVarChange);
        
        label.appendChild(checkbox);
        label.appendChild(document.createTextNode(col));
        indepVarsContainer.appendChild(label);
    });
}

function updateCheckboxSelection() {
    const checkboxes = indepVarsContainer.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(cb => {
        cb.checked = independentVars.includes(cb.value);
        cb.disabled = cb.value === targetVarSelect.value;
    });
}

function handleIndepVarChange(e) {
    const val = e.target.value;
    if (e.target.checked) {
        if (!independentVars.includes(val)) independentVars.push(val);
    } else {
        independentVars = independentVars.filter(v => v !== val);
    }
    
    if (independentVars.length === 0) {
        // Fallback to avoid empty
        e.target.checked = true;
        independentVars.push(val);
    }
    updateRegression();
}

// Math/Regression Logic
function updateRegression() {
    targetVar = targetVarSelect.value;
    updateCheckboxSelection();
    
    if (independentVars.includes(targetVar)) {
        independentVars = independentVars.filter(v => v !== targetVar);
        if (independentVars.length === 0) {
            const otherCol = columns.find(c => c !== targetVar);
            if(otherCol) independentVars.push(otherCol);
        }
        updateCheckboxSelection();
    }
    
    if (independentVars.length === 0) return;

    isMultiple = independentVars.length > 1;
    toggle3dBtn.style.display = (independentVars.length === 2) ? 'block' : 'none';
    if(independentVars.length !== 2) is3D = false;

    calculateOLS();
}

function calculateOLS() {
    try {
        const y = dataset.map(row => row[targetVar]);
        
        const X = dataset.map(row => {
            const rowArr = [1];
            independentVars.forEach(col => rowArr.push(row[col]));
            return rowArr;
        });

        // OLS: ß = (X^T * X)^-1 * X^T * y
        const X_mat = math.matrix(X);
        const y_mat = math.matrix(y);
        const X_t = math.transpose(X_mat);
        const X_t_X = math.multiply(X_t, X_mat);
        const X_t_X_inv = math.inv(X_t_X);
        const X_t_y = math.multiply(X_t, y_mat);
        const beta = math.multiply(X_t_X_inv, X_t_y);

        modelCoefs = beta.toArray();
        userCoefs = [...modelCoefs];

        buildSliders();
        updatePlots();
    } catch (e) {
        console.error("Matrix inversion failed:", e);
        alert("Regression failed, variables might be collinear or invalid.");
    }
}

function buildSliders() {
    modelControlsDiv.style.display = 'block';
    slidersContainer.innerHTML = '';
    
    createSlider(0, 'Intercept', userCoefs[0]);
    independentVars.forEach((v, i) => {
        createSlider(i + 1, v + ' Coef', userCoefs[i + 1]);
    });
}

function createSlider(index, labelText, initialValue) {
    let range = Math.max(Math.abs(initialValue) * 2, 10);
    if(range < 0.1) range = 1;
    
    const min = initialValue - range;
    const max = initialValue + range;
    
    const wrapper = document.createElement('div');
    wrapper.className = 'slider-container';
    
    const header = document.createElement('div');
    header.className = 'slider-header';
    const label = document.createElement('span');
    label.textContent = labelText;
    const valDisplay = document.createElement('span');
    valDisplay.className = 'slider-val';
    valDisplay.textContent = initialValue.toFixed(4);
    
    header.appendChild(label);
    header.appendChild(valDisplay);
    
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = min;
    slider.max = max;
    slider.step = (max - min) / 1000;
    slider.value = initialValue;
    
    slider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        valDisplay.textContent = val.toFixed(4);
        userCoefs[index] = val;
        updatePlots();
    });
    
    wrapper.appendChild(header);
    wrapper.appendChild(slider);
    slidersContainer.appendChild(wrapper);
}

function updateSliders() {
    const wrappers = slidersContainer.querySelectorAll('.slider-container');
    wrappers.forEach((w, i) => {
        const slider = w.querySelector('input');
        const display = w.querySelector('.slider-val');
        slider.value = userCoefs[i];
        display.textContent = userCoefs[i].toFixed(4);
    });
}

function predict(row, coefs) {
    let pred = coefs[0];
    independentVars.forEach((v, i) => {
        pred += coefs[i+1] * row[v];
    });
    return pred;
}

function updatePlots() {
    const y = dataset.map(row => row[targetVar]);
    const preds = dataset.map(row => predict(row, userCoefs));
    const residuals = y.map((yi, i) => yi - preds[i]);
    
    const yMean = math.mean(y);
    const ssTot = y.reduce((acc, yi) => acc + Math.pow(yi - yMean, 2), 0);
    const ssRes = residuals.reduce((acc, res) => acc + Math.pow(res, 2), 0);
    
    let r2 = 1 - (ssRes / ssTot);
    if (ssTot === 0) r2 = 0;
    const mse = ssRes / y.length;
    
    r2ScoreDisplay.textContent = r2.toFixed(4);
    mseScoreDisplay.textContent = mse.toFixed(2);
    
    let eq = `Y = ${userCoefs[0].toFixed(2)}`;
    independentVars.forEach((v, i) => {
        let sign = userCoefs[i+1] >= 0 ? '+' : '-';
        eq += ` ${sign} ${Math.abs(userCoefs[i+1]).toFixed(2)}(${v})`;
    });
    equationDisplay.textContent = eq;

    const layoutConfig = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#94a3b8', family: 'Inter' },
        margin: { t: 10, r: 10, b: 40, l: 50 },
        xaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' },
        yaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' },
    };

    if (!isMultiple) {
        // 2D Simple Linear
        const xData = dataset.map(row => row[independentVars[0]]);
        const minX = Math.min(...xData);
        const maxX = Math.max(...xData);
        const lineX = [minX, maxX];
        const lineY = [
            userCoefs[0] + userCoefs[1] * minX,
            userCoefs[0] + userCoefs[1] * maxX
        ];

        const trace1 = {
            x: xData, y: y, mode: 'markers', type: 'scatter',
            name: 'Data', marker: { color: '#3b82f6', size: 6, opacity: 0.8 }
        };
        const trace2 = {
            x: lineX, y: lineY, mode: 'lines', type: 'scatter',
            name: 'Fit', line: { color: '#ef4444', width: 3 }
        };

        Plotly.newPlot('plot-main', [trace1, trace2], {
            ...layoutConfig,
            xaxis: { ...layoutConfig.xaxis, title: independentVars[0] },
            yaxis: { ...layoutConfig.yaxis, title: targetVar },
            showlegend: false
        }, {responsive: true});

    } else if (independentVars.length === 2 && is3D) {
        const x1 = dataset.map(row => row[independentVars[0]]);
        const x2 = dataset.map(row => row[independentVars[1]]);
        
        const trace1 = {
            x: x1, y: x2, z: y, mode: 'markers', type: 'scatter3d',
            marker: { color: '#3b82f6', size: 4, opacity: 0.8 }
        };

        const steps = 10;
        const x1min = Math.min(...x1), x1max = Math.max(...x1);
        const x2min = Math.min(...x2), x2max = Math.max(...x2);
        const x1vals = [], x2vals = [], zvals = [];
        
        for(let i=0; i<steps; i++) {
            let cx1 = x1min + (x1max - x1min)*(i/(steps-1));
            let zrow = [];
            for(let j=0; j<steps; j++) {
                let cx2 = x2min + (x2max - x2min)*(j/(steps-1));
                if(i===0) x2vals.push(cx2);
                zrow.push(userCoefs[0] + userCoefs[1]*cx1 + userCoefs[2]*cx2);
            }
            x1vals.push(cx1);
            zvals.push(zrow);
        }

        const trace2 = {
            x: x1vals, y: x2vals, z: zvals, type: 'surface',
            colorscale: 'Reds', opacity: 0.6, showscale: false
        };

        Plotly.newPlot('plot-main', [trace1, trace2], {
            ...layoutConfig,
            margin: { t: 0, r: 0, b: 0, l: 0 },
            scene: {
                xaxis: { title: independentVars[0], gridcolor: '#333' },
                yaxis: { title: independentVars[1], gridcolor: '#333' },
                zaxis: { title: targetVar, gridcolor: '#333' },
                bgcolor: 'rgba(0,0,0,0)'
            },
            showlegend: false
        }, {responsive: true});

    } else {
        // Predicted vs Actual
        const trace = {
            x: preds, y: y, mode: 'markers', type: 'scatter',
            name: 'Data', marker: { color: '#3b82f6', size: 6, opacity: 0.8 }
        };
        const minVal = Math.min(Math.min(...preds), Math.min(...y));
        const maxVal = Math.max(Math.max(...preds), Math.max(...y));
        const lineTrace = {
            x: [minVal, maxVal], y: [minVal, maxVal], mode: 'lines', type: 'scatter',
            name: 'Perfect Fit', line: { color: '#ef4444', dash: 'dash' }
        };
        Plotly.newPlot('plot-main', [trace, lineTrace], {
            ...layoutConfig,
            xaxis: { ...layoutConfig.xaxis, title: 'Predicted ' + targetVar },
            yaxis: { ...layoutConfig.yaxis, title: 'Actual ' + targetVar },
            showlegend: false
        }, {responsive: true});
    }

    // Residuals Plot
    const resTrace = {
        x: preds, y: residuals, mode: 'markers', type: 'scatter',
        marker: { color: '#10b981', size: 5, opacity: 0.8 }
    };
    const zeroLine = {
        x: [Math.min(...preds), Math.max(...preds)], y: [0, 0], mode: 'lines', type: 'scatter',
        line: { color: 'rgba(255,255,255,0.3)', dash: 'dot', width: 2 }
    };
    
    Plotly.newPlot('plot-residuals', [resTrace, zeroLine], {
        ...layoutConfig,
        xaxis: { ...layoutConfig.xaxis, title: 'Predicted values' },
        yaxis: { ...layoutConfig.yaxis, title: 'Residuals' },
        showlegend: false
    }, {responsive: true});
}

window.addEventListener('resize', () => {
    if(dataset.length > 0) {
        Plotly.Plots.resize('plot-main');
        Plotly.Plots.resize('plot-residuals');
    }
});
