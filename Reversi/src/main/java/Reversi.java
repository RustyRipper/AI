import java.util.Random;
import java.util.Scanner;

public class Reversi {
    public char[][] board;
    public char currentPlayer;

    public Reversi() {
        board = new char[8][8];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                board[i][j] = ' ';
            }
        }
        board[3][3] = 'W';
        board[3][4] = 'B';
        board[4][3] = 'B';
        board[4][4] = 'W';
        currentPlayer = 'B';
    }

    public Reversi(char[][] board, char currentPlayer) {

        this.board = board;
        this.currentPlayer = currentPlayer;

    }

    public void startGame() throws InterruptedException {
        Scanner scanner = new Scanner(System.in);
        boolean gameOver = false;

        System.out.println("Gra Reversi rozpoczyna się!");
        displayBoard();
        Random random = new Random();
        ReversiMiniMax reversiMiniMax = new ReversiMiniMax('B');
        ReversiMiniMax reversiMiniMaxW = new ReversiMiniMax('W');
        while (!gameOver) {

            //System.out.println("Wprowadź ruch (np. '2 3'): ");
            //int row = scanner.nextInt();
            //int col = scanner.nextInt();
            int row = 0;
            int col = 0;
            if (currentPlayer == 'B') {
                ReversiMiniMax.MoveResult moveResult =
                        reversiMiniMax.minimaxalphabeta(new Reversi(Main.cloneCharArray(board), currentPlayer), 'B', 7, Integer.MIN_VALUE, Integer.MAX_VALUE);
                row = moveResult.row;
                col = moveResult.col;
                System.out.println("SCORE = " + moveResult.score);
            } else {
                ReversiMiniMax.MoveResult moveResult =
                        reversiMiniMaxW.minimaxalphabeta(new Reversi(Main.cloneCharArray(board), currentPlayer), 'W', 8, Integer.MIN_VALUE, Integer.MAX_VALUE);
                row = moveResult.row;
                col = moveResult.col;
                System.out.println("SCORE2 = " + moveResult.score);
            }


            if (row >= 0 && col >= 0 && isValidMove(row, col, currentPlayer)) {
                System.out.println("Ruch gracza " + currentPlayer);
                //Thread.sleep(2000);
                makeMove(row, col, currentPlayer);
                displayBoard();
                gameOver = handleNoMoveAvailable(false);

            } else if (row < 0 || col < 0) {
                switchPlayer();
            } else {
                System.out.println("Nieprawidłowy ruch, spróbuj ponownie.");
            }
        }
    }

    private void switchPlayer() {
        currentPlayer = (currentPlayer == 'B') ? 'W' : 'B';
    }

    private int countScore(char playerColor) {
        int score = 0;
        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 8; col++) {
                if (board[row][col] == playerColor) {
                    score++;
                }
            }
        }
        return score;
    }

    public void makeMove(int row, int col, char player) {
        if (!isValidMove(row, col, player)) {
            System.out.println("Nieprawidłowy ruch!");
            return;
        }
        board[row][col] = player;
        flipOpponentPieces(row, col);
        switchPlayer();

    }

    public boolean isValidMove(int row, int col, char currentPlayer) {
        if (board[row][col] != ' ') {
            return false;
        }
        char opponentColor = (currentPlayer == 'B') ? 'W' : 'B';
        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                if (dr == 0 && dc == 0) {
                    continue;
                }
                int r = row + dr;
                int c = col + dc;
                boolean foundOpponent = false;
                while (r >= 0 && r < 8 && c >= 0 && c < 8 && board[r][c] == opponentColor) {
                    r += dr;
                    c += dc;
                    foundOpponent = true;
                }
                if (r >= 0 && r < 8 && c >= 0 && c < 8 && board[r][c] == currentPlayer && foundOpponent) {
                    return true;
                }
            }
        }
        return false;
    }

    public void flipOpponentPieces(int row, int col) {
        char opponentColor = (currentPlayer == 'B') ? 'W' : 'B';
        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                if (dr == 0 && dc == 0) {
                    continue;
                }
                int r = row + dr;
                int c = col + dc;
                boolean foundOpponent = false;
                while (r >= 0 && r < 8 && c >= 0 && c < 8 && board[r][c] == opponentColor) {
                    r += dr;
                    c += dc;
                    foundOpponent = true;
                }
                if (r >= 0 && r < 8 && c >= 0 && c < 8 && board[r][c] == currentPlayer && foundOpponent) {
                    int flipRow = row + dr;
                    int flipCol = col + dc;
                    while (flipRow != r || flipCol != c) {
                        board[flipRow][flipCol] = currentPlayer;
                        flipRow += dr;
                        flipCol += dc;
                    }
                }
            }
        }
    }

    public boolean noValidMove(char player) {
        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 8; col++) {
                if (isValidMove(row, col, player)) {
                    return false;
                }
            }
        }
        return true;
    }

    public boolean handleNoMoveAvailable(boolean silent) {
        if (noValidMove(currentPlayer)) {
            if (!silent) {
                System.out.println("Gracz " + currentPlayer + " nie ma możliwego ruchu.");
            }

            switchPlayer();
            if (noValidMove(currentPlayer)) {
                if (!silent) {
                    System.out.println("Obaj gracze nie mają możliwych ruchów. Koniec gry.");
                    int blackScore = countScore('B');
                    int whiteScore = countScore('W');
                    System.out.println("Wynik: Gracz B - " + blackScore + " punktów, Gracz W - " + whiteScore + " punktów");
                    if (blackScore > whiteScore) {
                        System.out.println("Gracz B wygrał!");
                    } else if (blackScore < whiteScore) {
                        System.out.println("Gracz W wygrał!");
                    } else {
                        System.out.println("Remis!");
                    }
                }
                return true;
            }
        }
        return false;
    }

    public void displayBoard() {
        System.out.println("  0 1 2 3 4 5 6 7");
        for (int row = 0; row < 8; row++) {
            System.out.print(row + " ");
            for (int col = 0; col < 8; col++) {
                System.out.print(board[row][col] + " ");
            }
            System.out.println();
        }
    }
}
