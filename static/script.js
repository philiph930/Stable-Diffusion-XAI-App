/*
document.getElementById("PreLoaderBar").style.display = "block";  // Show loading indicator

fetch('/your-url', { method: 'GET' })
    .then(response => response.json())
    .then(data => {
        // Process the response data
        console.log("Data received:", data);
        document.getElementById("PreLoaderBar").style.display = "none";  // Hide loading indicator
    })
    .catch(error => {
        // Handle errors
        console.error("Error:", error);
        document.getElementById("PreLoaderBar").style.display = "none";  // Hide loading indicator even in case of an error
    });
*/

/*
window.onload = function () {
    console.log("Page fully loaded");
    document.getElementById("PreLoaderBar").style.display = "none";
};

document.onreadystatechange = function () {
    if (document.readyState === "complete") {
        console.log(document.readyState);
        document.getElementById("PreLoaderBar").style.display = "none";
    }

    else {
        console.log("loading");
        document.getElementById("PreLoaderBar").style.display = "block";
    }
};
*/