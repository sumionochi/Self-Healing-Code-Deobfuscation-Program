<?php
// This function is never called
function unused_function($a, $b) {
    return $a + $b;
}

$x = 10;  // Used variable
$y = 20;  // Unused variable

if (true) {
    echo "This will always run\\n";
} else {
    echo "This will never run\\n";  // Dead branch
}

echo "Before return\\n";
if ($x > 5) {
    echo "Will run\\n";
    exit();  // Makes code below unreachable
    echo "This is unreachable\\n";  // Unreachable code
}

echo "This will never execute\\n";  // Unreachable due to exit
?>