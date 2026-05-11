function updateSpeed(value) {
    console.log("กำลังส่งความเร็ว: ",value);

    fetch('/api/set_speed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speed: value })
    });
}

const slider = document.getElementById('speedSlider');
slider.oninput = function() {
    updateSpeed(this.value);
};

function stopRobot() {
    alert(" สั่งหยุดหุ่นยนต์");
    fetch('/api/stop', { method: 'POST'});
}