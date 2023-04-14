public class Main {
    public static void main(String[] args) throws InterruptedException {
        System.out.println("Hello world!");
        Reversi reversi = new Reversi();
        reversi.startGame();

    }

    public static char[][] cloneCharArray(char[][] original) {

        if (original == null) {
            return null;
        }

        int rows = original.length;
        int cols = original[0].length;

        char[][] clone = new char[rows][cols];

        for (int i = 0; i < rows; i++) {
            System.arraycopy(original[i], 0, clone[i], 0, cols);
        }
        return clone;
    }
}