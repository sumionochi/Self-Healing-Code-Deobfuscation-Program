<?php
// Obfuscated PHP Code Example

// Function 'a' multiplies two numbers, adds 3, and adjusts based on even/odd.
function a($b, $c) {
    $d = $b * $c;
    $e = $d + 3;
    if ($e % 2 == 0) {
        $f = $e / 2;
    } else {
        $f = ($e + 1) / 2;
    }
    return $f;
}

// Function 'g' takes a string, XORs each character with 33, and builds a new string.
function g($h) {
    $i = "";
    for ($j = 0; $j < strlen($h); $j++) {
        $k = ord($h[$j]);
        $l = $k ^ 33;
        $i .= chr($l);
    }
    return $i;
}

// Function 'm' reverses each character in the string individually.
function m($n) {
    $o = "";
    $p = str_split($n);
    foreach ($p as $q) {
        $o .= strrev($q);
    }
    return $o;
}

// Class 'r' uses the above functions to process a value.
class r {
    private $s;
    public function __construct($t) {
        $this->s = $t;
    }
    public function u($v) {
        $w = a($this->s, $v);
        $x = g((string)$w);
        $y = m($x);
        return $y;
    }
}

// Function 'z' increments the ASCII code of each character in the string.
function z($aa) {
    $ab = "";
    for ($ac = 0; $ac < strlen($aa); $ac++) {
        $ad = ord($aa[$ac]);
        $ab .= chr($ad + 1);
    }
    return $ab;
}

// Main execution:
$ae = new r(7);
$af = $ae->u(9);
$ag = z($af);
echo $ag;
?>
