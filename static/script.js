/* ================= GLOBAL VARIABLES ================= */

let chartInstance = null;
let radarInstance = null;
let compareChartInstance = null;

let lastMaterials = [];
let lastPayload = null;
let databaseMaterials = [];

let trendChartInstance = null;
let categoryChartInstance = null;
let materialChart, costChart;

/**Enterprise JS */


function loadEnterpriseDashboard(category=null) {

    let url = "/dashboard_data";
    if(category) url += `?material=${category}`;

    fetch(url)
    .then(res => res.json())
    .then(data => {

        document.getElementById("totalReports").innerText = data.total_reports;
        document.getElementById("topMaterial").innerText = data.top_material;
        document.getElementById("avgEco").innerText = data.avg_eco;
        document.getElementById("co2Reduction").innerText = data.co2_reduction;
        document.getElementById("costSavings").innerText = "₹ " + data.cost_savings;
        document.getElementById("betterPlastic").innerText = data.better_than_plastic + "%";

        drawMaterialChart(data);
        drawCostChart(data);
    });
}


function drawMaterialChart(data) {

    const ctx = document.getElementById("materialChart").getContext("2d");
    if(materialChart) materialChart.destroy();

    materialChart = new Chart(ctx,{
        type:'doughnut',
        data:{
            labels:data.materials,
            datasets:[{data:data.material_counts}]
        },
        options:{
            onClick:(evt, elements)=>{
                if(elements.length>0){
                    const index = elements[0].index;
                    const selected = data.materials[index];
                    loadEnterpriseDashboard(selected);
                }
            }
        }
    });
}

function drawCostChart(data){

    const ctx = document.getElementById("costChart").getContext("2d");
    if(costChart) costChart.destroy();

    costChart = new Chart(ctx,{
        type:'line',
        data:{
            labels:data.cumulative_cost.map((_,i)=>`Report ${i+1}`),
            datasets:[
                {label:'Cumulative INR Saved', data:data.cumulative_cost},
                {label:'Cumulative CO₂ Avoided', data:data.cumulative_co2}
            ]
        }
    });
}

function animateCounter(id, value) {
    let start = 0;
    const duration = 1000;
    const increment = value / 50;

    const interval = setInterval(()=>{
        start += increment;
        if(start >= value){
            document.getElementById(id).innerText = value;
            clearInterval(interval);
        } else {
            document.getElementById(id).innerText = start.toFixed(1);
        }
    }, duration/50);
}

function applyFilter() {

    const start = document.getElementById("startDate").value;
    const end = document.getElementById("endDate").value;

    let url = `/dashboard_data?start=${start}&end=${end}`;

    fetch(url)
    .then(res => res.json())
    .then(data => {
        document.getElementById("totalReports").innerText = data.total_reports;
        document.getElementById("avgEco").innerText = data.avg_eco;
        document.getElementById("co2Reduction").innerText = data.co2_reduction;
        document.getElementById("costSavings").innerText = "₹ " + data.cost_savings;
        document.getElementById("betterPlastic").innerText = data.better_than_plastic + "%";

        drawMaterialChart(data);
        drawCostChart(data);
    });
}


async function exportPDF() {

    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF();

    pdf.text("EcoPack Enterprise Sustainability Report", 10, 10);

    pdf.text(document.getElementById("aiInsight").innerText, 10, 20);

    const canvas = document.getElementById("materialChart");
    const img = canvas.toDataURL("image/png");

    pdf.addImage(img, "PNG", 10, 40, 180, 100);

    pdf.save("EcoPack_Enterprise_Report.pdf");
}

/**Enterprise JS */



function animateValue(id, value) {
    const el = document.getElementById(id);
    let start = 0;
    const duration = 800;
    const stepTime = 20;
    const increment = value / (duration / stepTime);

    const timer = setInterval(() => {
        start += increment;
        if (start >= value) {
            el.innerText = value;
            clearInterval(timer);
        } else {
            el.innerText = start.toFixed(1);
        }
    }, stepTime);
}


function loadDashboard() {

    fetch("/dashboard_data")
        .then(res => res.json())
        .then(data => {

            if (data.error) {
                console.error(data.error);
                return;
            }

animateValue("biTotalReports", data.total_reports);
document.getElementById("biTopMaterial").innerText = data.top_material;
animateValue("biAvgEco", data.avg_eco);
animateValue("biAvgCO2", data.avg_co2);
animateValue("biAvgCost", data.avg_cost);

            drawTrendChart(data);
            drawCategoryChart(data);

        })
        .catch(err => console.error("Dashboard error:", err));
}

function drawTrendChart(data) {

    const ctx = document.getElementById("trendChart").getContext("2d");

    if (trendChartInstance) trendChartInstance.destroy();

    trendChartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: data.eco_trend.map((_, i) => `Report ${i+1}`),
            datasets: [
                { label: "Eco Score", data: data.eco_trend },
                { label: "CO₂", data: data.co2_trend },
                { label: "Cost", data: data.cost_trend }
            ]
        },
        options: { responsive: true }
    });
}

function drawCategoryChart(data) {

    const ctx = document.getElementById("categoryChart").getContext("2d");

    if (categoryChartInstance) categoryChartInstance.destroy();

    categoryChartInstance = new Chart(ctx, {
        type: "pie",
        data: {
            labels: data.categories,
            datasets: [{
                data: data.category_counts
            }]
        },
        options: { responsive: true }
    });
}

function saveReport() {
    if (!window.lastBestMaterial) {
        alert("Please generate recommendation first!");
        return;
    }

    fetch("/save-report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            product_category: document.getElementById("category").value,
            selected_material: window.lastBestMaterial.material,
            eco_score: window.lastBestMaterial.eco_score,
            predicted_co2: window.lastBestMaterial.predicted_co2,
            predicted_cost: window.lastBestMaterial.predicted_cost
        })
    })
    .then(res => res.json())
    .then(data => {
        alert("Report saved successfully!");
        loadEnterpriseDashboard();
    })
    .catch(err => {
        console.error(err);
        alert("Error saving report");
    });
}


/* =====================================================
   INITIAL LOAD
===================================================== */

function initApp() {
    loadDatabaseMaterials();
    initializeSelect2();
    initializeDarkMode();
    initializeSideMenu();
    loadDashboard();
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initApp);
} else {
    initApp();
}


/* =====================================================
   SELECT2 INIT
===================================================== */

function initializeSelect2() {
    if (window.jQuery && $('#dbMaterialSelect').length) {
        $('#dbMaterialSelect').select2({
            placeholder: "Search material...",
            width: "100%"
        });
    }
}

/* =====================================================
   LOAD DATABASE MATERIALS
===================================================== */

function loadDatabaseMaterials() {

    fetch("/materials")
        .then(res => res.json())
        .then(data => {

            databaseMaterials = data;
            const dropdown = document.getElementById("dbMaterialSelect");
            dropdown.innerHTML = "";

            data.forEach(mat => {
                const option = document.createElement("option");
                option.value = mat.material;
                option.textContent = mat.material;
                dropdown.appendChild(option);
            });

            // Proper Select2 re-init
            if (window.jQuery) {
                $('#dbMaterialSelect').select2('destroy');
                $('#dbMaterialSelect').select2({
                    placeholder: "Search material...",
                    width: "100%"
                });
            }
        })
        .catch(err => console.error("Database load error:", err));
}


/* =====================================================
   SAFE LEVEL MAPPER
===================================================== */

function levelToNum(level) {
    const map = { Low: 3, Medium: 6, High: 9 };
    return map[level] || 3;
}

/* =====================================================
   MAIN AI FETCH
===================================================== */

function getRecommendations() {

    lastPayload = {
        product_category: document.getElementById("category").value || "",
        strength_score: levelToNum(document.getElementById("strength_score").value),
        weight_capacity_kg: Number(document.getElementById("weight_capacity_kg").value) || 0,
        biodegradability_score: levelToNum(document.getElementById("biodegradability_score").value),
        recyclability_percent: Number(document.getElementById("recyclability_percent").value) || 0,
        moisture_resistance: levelToNum(document.getElementById("moisture_resistance").value),
        heat_resistance: levelToNum(document.getElementById("heat_resistance").value)
    };

    document.getElementById("loader").classList.remove("hidden");

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(lastPayload)
    })
    .then(res => res.json())
    .then(data => {

        if (data.error) {
            alert(data.error);
            return;
        }

        lastMaterials = data.recommended_materials || [];
// 🔥 SAVE FOR RECOMMENDATION DASHBOARD
localStorage.setItem(
    "recommendationData",
    JSON.stringify(lastMaterials)
);

        if (!lastMaterials.length) {
            alert("No materials returned from AI.");
            return;
        }

        populateTable(lastMaterials);
        populateCompareDropdown(lastMaterials);
        drawBarChart(lastMaterials);
        updateKPIs(lastMaterials);
        showInsights(lastMaterials);
        showRejected(data.rejected_materials || []);
        showAIReason(lastMaterials[0]);
        drawRadar(lastMaterials[0]);
    })
    .catch(err => console.error("Prediction error:", err))
    .finally(() => {
        document.getElementById("loader").classList.add("hidden");
    });
}

/* =====================================================
   TABLE
===================================================== */

function populateTable(materials) {

    const tbody = document.getElementById("resultsBody");
    tbody.innerHTML = "";

    const minCost = Math.min(...materials.map(m => m.predicted_cost));
    const minCO2 = Math.min(...materials.map(m => m.predicted_co2));

    materials.forEach((m, index) => {

        const row = document.createElement("tr");

        row.innerHTML = `
            <td>${index + 1}</td>
            <td>${m.material}</td>
            <td>${m.eco_score}</td>
            <td>${m.predicted_co2}</td>
            <td>${m.predicted_cost}</td>
            <td>${m.predicted_cost === minCost ? "Lowest Cost" : "-"}</td>
            <td>${m.predicted_co2 === minCO2 ? "Lowest CO₂" : "-"}</td>
        `;

        tbody.appendChild(row);
    });
}

/* =====================================================
   DROPDOWN (AI Materials)
===================================================== */

function populateCompareDropdown(materials) {

    const dropdown = document.getElementById("compareMaterial");
    dropdown.innerHTML = "";

    materials.forEach(m => {
        const option = document.createElement("option");
        option.value = m.material;
        option.textContent = m.material;
        dropdown.appendChild(option);
    });
}

/* =====================================================
   COMPARE AI MATERIAL (RADAR SWITCH)
===================================================== */

function compareMaterial() {

    if (!lastMaterials.length) return;

    const selectedName = document.getElementById("compareMaterial").value;

    const selectedMaterial = lastMaterials.find(
        m => m.material === selectedName
    );

    if (!selectedMaterial) return;

    const bestMaterial = lastMaterials[0];

    drawLineComparison(bestMaterial, selectedMaterial);
}
function drawLineComparison(mat1, mat2) {

    const ctx = document.getElementById("radarChart").getContext("2d");

    if (radarInstance) radarInstance.destroy();

    radarInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: ["CO₂", "Cost", "Eco Score"],
            datasets: [
                {
                    label: mat1.material,
                    data: [
                        mat1.predicted_co2,
                        mat1.predicted_cost,
                        mat1.eco_score
                    ],
                    tension: 0.3
                },
                {
                    label: mat2.material,
                    data: [
                        mat2.predicted_co2,
                        mat2.predicted_cost,
                        mat2.eco_score
                    ],
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}


/* =====================================================
   DATABASE COMPARISON
===================================================== */

function compareWithDatabase() {

    if (!lastMaterials.length) {
        alert("Please generate AI recommendation first.");
        return;
    }

    const bestMaterial = lastMaterials[0];
    const selectedName = document.getElementById("dbMaterialSelect").value;

    const selectedMaterial = databaseMaterials.find(
        m => m.material === selectedName
    );

    if (!selectedMaterial) return;

    drawComparisonChart(bestMaterial, selectedMaterial);
}

function drawComparisonChart(mat1, mat2) {

    const ctx = document.getElementById("compareChart").getContext("2d");

    if (compareChartInstance) compareChartInstance.destroy();

    compareChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["CO₂", "Cost", "Eco Score"],
            datasets: [
                {
                    label: mat1.material,
                    data: [mat1.predicted_co2, mat1.predicted_cost, mat1.eco_score]
                },
                {
                    label: mat2.material,
                    data: [mat2.predicted_co2, mat2.predicted_cost, mat2.eco_score]
                }
            ]
        },
        options: {
            responsive: true,
            scales: { y: { beginAtZero: true } }
        }
    });
}

/* =====================================================
   BAR CHART
===================================================== */

function drawBarChart(materials) {

    const ctx = document.getElementById("ecoChart").getContext("2d");

    if (chartInstance) chartInstance.destroy();

    chartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: materials.map(m => m.material),
            datasets: [
                { label: "CO₂ Emissions", data: materials.map(m => m.predicted_co2) },
                { label: "Cost", data: materials.map(m => m.predicted_cost) }
            ]
        },
        options: {
            responsive: true,
            scales: { y: { beginAtZero: true } }
        }
    });
}

/* =====================================================
   RADAR CHART
===================================================== */

function drawRadar(material) {

    const ctx = document.getElementById("radarChart").getContext("2d");

    if (radarInstance) radarInstance.destroy();

    radarInstance = new Chart(ctx, {
        type: "radar",
        data: {
            labels: ["CO₂", "Cost", "Eco Score"],
            datasets: [{
                label: material.material,
                data: [
                    material.predicted_co2,
                    material.predicted_cost,
                    material.eco_score
                ]
            }]
        },
        options: {
            responsive: true,
            scales: { r: { beginAtZero: true } }
        }
    });
}

/* =====================================================
   KPIs
===================================================== */

function updateKPIs(materials) {

    const avgEco = (
        materials.reduce((s, m) => s + m.eco_score, 0) / materials.length
    ).toFixed(2);

    const avgCO2 = (
        materials.reduce((s, m) => s + m.predicted_co2, 0) / materials.length
    ).toFixed(2);

    const avgCost = (
        materials.reduce((s, m) => s + m.predicted_cost, 0) / materials.length
    ).toFixed(2);

    document.getElementById("avgEcoScore").innerText = avgEco;
    document.getElementById("avgCO2").innerText = avgCO2;
    document.getElementById("avgCost").innerText = avgCost;
}

/* =====================================================
   AI REASON
===================================================== */

function showAIReason(best) {

    document.getElementById("aiText").innerText =
        `${best.material} is recommended as the most sustainable option 
        with Eco Score ${best.eco_score}, CO₂ ${best.predicted_co2}, 
        and Cost Index ${best.predicted_cost}.`;
}

/* =====================================================
   REJECTED
===================================================== */

function showRejected(rejected) {

    const list = document.getElementById("rejectedList");
    list.innerHTML = "";

    console.log("Rejected received:", rejected);

    if (!rejected || rejected.length === 0) {
        list.innerHTML = "<li>No rejected materials 🎉</li>";
        return;
    }

    rejected.slice(0, 10).forEach(r => {

        const li = document.createElement("li");

        const reasons = Array.isArray(r.reasons)
            ? r.reasons.join(", ")
            : r.reasons || "No reason provided";

        li.innerText = `${r.material}: ${reasons}`;

        list.appendChild(li);
    });
}

/* =====================================================
   INSIGHTS (Lowest CO2, Lowest Cost, Best Overall)
===================================================== */
function showInsights(materials) {

    if (!materials || materials.length === 0) return;

    const lowestCO2 = materials.reduce((min, m) =>
        m.predicted_co2 < min.predicted_co2 ? m : min
    );

    const lowestCost = materials.reduce((min, m) =>
        m.predicted_cost < min.predicted_cost ? m : min
    );

    const bestOverall = materials.reduce((max, m) =>
        m.eco_score > max.eco_score ? m : max
    );

    document.getElementById("insightCO2").innerText =
        `${lowestCO2.material} (${lowestCO2.predicted_co2})`;

    document.getElementById("insightCost").innerText =
        `${lowestCost.material} (${lowestCost.predicted_cost})`;

    document.getElementById("insightBest").innerText =
        `${bestOverall.material} (Eco: ${bestOverall.eco_score})`;
    window.lastBestMaterial = bestOverall;


    }


/* =====================================================
   DARK MODE
===================================================== */

function initializeDarkMode() {

    const darkBtn = document.getElementById("darkToggle");
    if (!darkBtn) return;

    function updateIcon() {
        darkBtn.textContent =
            document.body.classList.contains("dark") ? "☀️" : "🌙";
    }

    darkBtn.onclick = function () {
        document.body.classList.toggle("dark");

        localStorage.setItem(
            "theme",
            document.body.classList.contains("dark") ? "dark" : "light"
        );

        updateIcon();
    };

    if (localStorage.getItem("theme") === "dark") {
        document.body.classList.add("dark");
    }

    updateIcon();
}


/* =====================================================
   SIDE MENU
===================================================== */

function initializeSideMenu() {

    const menuBtn = document.getElementById("menuToggle");
    const sideMenu = document.getElementById("sideMenu");

    if (!menuBtn || !sideMenu) return;

    menuBtn.onclick = function () {
        sideMenu.classList.toggle("active");
    };

    sideMenu.querySelectorAll("a").forEach(link => {
        link.onclick = function () {
            sideMenu.classList.remove("active");
        };
    });
}

/* =====================================================
   EXPORT CSV
===================================================== */
function exportCSV() {

    if (!lastMaterials.length) {
        alert("No data to export.");
        return;
    }

    let csv = "=== AI RECOMMENDED MATERIALS ===\n";
    csv += "Rank,Material,Eco Score,CO2,Cost\n";

    lastMaterials.forEach((m, index) => {
        csv += `${index + 1},${m.material},${m.eco_score},${m.predicted_co2},${m.predicted_cost}\n`;
    });

    csv += "\n=== REJECTED MATERIALS ===\n";
    csv += "Material,Reasons\n";

    const rejectedList = document.querySelectorAll("#rejectedList li");

    if (rejectedList.length === 0) {
        csv += "None,None\n";
    } else {
        rejectedList.forEach(li => {
            const text = li.innerText.replace(",", " -");
            csv += `${text}\n`;
        });
    }

    const blob = new Blob([csv], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "EcoPack_AI_Full_Report.csv";
    a.click();

    window.URL.revokeObjectURL(url);
}

/* =====================================================
   EXPORT PDF (Simple Version)
===================================================== */

async function exportPDF() {

    if (!lastMaterials.length) {
        alert("No data to export.");
        return;
    }

    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF("p", "mm", "a4");

    let y = 10;

    pdf.setFontSize(16);
    pdf.text("EcoPack AI Report", 10, y);
    y += 10;

    /* =============================
       AI RECOMMENDED MATERIALS
    ============================== */

    pdf.setFontSize(12);
    pdf.text("AI Recommended Materials:", 10, y);
    y += 8;

    lastMaterials.forEach((m, index) => {
        pdf.text(
            `${index + 1}. ${m.material} | Eco: ${m.eco_score} | CO2: ${m.predicted_co2} | Cost: ${m.predicted_cost}`,
            10,
            y
        );
        y += 6;
    });

    y += 8;

    /* =============================
       REJECTED MATERIALS
    ============================== */

    pdf.text("Rejected Materials:", 10, y);
    y += 8;

    const rejectedList = document.querySelectorAll("#rejectedList li");

    if (rejectedList.length === 0) {
        pdf.text("None", 10, y);
        y += 6;
    } else {
        rejectedList.forEach(li => {
            pdf.text(li.innerText, 10, y);
            y += 6;
        });
    }

    /* =============================
       FORCE CHART UPDATE
    ============================== */

    if (chartInstance) chartInstance.update();
    if (compareChartInstance) compareChartInstance.update();
    if (radarInstance) radarInstance.update();

    // Wait 500ms to ensure charts are fully rendered
    await new Promise(resolve => setTimeout(resolve, 500));

    /* =============================
       CO2 vs COST CHART
    ============================== */

    if (chartInstance) {

        const ecoCanvas = document.getElementById("ecoChart");
        const ecoImage = ecoCanvas.toDataURL("image/png", 1.0);

        pdf.addPage();
        pdf.setFontSize(14);
        pdf.text("CO2 vs Cost Comparison", 10, 10);
        pdf.addImage(ecoImage, "PNG", 10, 20, 180, 100);
    }

    /* =============================
       DATABASE COMPARISON CHART
    ============================== */

    if (compareChartInstance) {

        const compareCanvas = document.getElementById("compareChart");
        const compareImage = compareCanvas.toDataURL("image/png", 1.0);

        pdf.addPage();
        pdf.setFontSize(14);
        pdf.text("Comparison With Database Material", 10, 10);
        pdf.addImage(compareImage, "PNG", 10, 20, 180, 100);
    }

    /* =============================
       AI MATERIAL COMPARISON
    ============================== */

    if (radarInstance) {

        const radarCanvas = document.getElementById("radarChart");
        const radarImage = radarCanvas.toDataURL("image/png", 1.0);

        pdf.addPage();
        pdf.setFontSize(14);
        pdf.text("AI Material Comparison", 10, 10);
        pdf.addImage(radarImage, "PNG", 10, 20, 180, 100);
    }

    pdf.save("EcoPack_AI_Full_Report.pdf");
}
