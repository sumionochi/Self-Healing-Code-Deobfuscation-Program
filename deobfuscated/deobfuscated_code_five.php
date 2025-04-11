<?php
// This function is never called
function unused_function($firstNumber, $secondNumber) {
    return $firstNumber + $secondNumber;
}

$usedVariable = 10;  // Used variable
$unusedVariable = 20;  // Unused variable

if (true) {
    echo "This will always run\\n";
} else {
    echo "/ Dead branch"
}

echo "Before return\\n";
if ($threshold > 5) {
    echo "Will run\\n";
    exit();  // Makes code below unreachable
    echo "This is unreachable\\n";  // Unreachable code
}

echo "This will never execute\\n";  // Unreachable due to exit
?>