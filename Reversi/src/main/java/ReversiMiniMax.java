import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;

public class ReversiMiniMax {

    char playerMove;

    public ReversiMiniMax(char player) {
        this.playerMove = player;
    }

    static class MoveResult {
        int score;
        int row;
        int col;
        int nodeVisited;

        long time;

        public MoveResult(int score, int row, int col) {
            this.score = score;
            this.row = row;
            this.col = col;
            this.nodeVisited = 0;
            this.time = Duration.between(Instant.now(), Instant.now()).toMillis();
        }

        public MoveResult(int score, int row, int col, int nodeVisited, long time) {
            this.score = score;
            this.row = row;
            this.col = col;
            this.nodeVisited = nodeVisited;
            this.time = time;
        }
    }

    public MoveResult minimax(Reversi reversi, char player, int depth, String mode) {
        Instant start = Instant.now();
        int nodesVisited = 0;

        if (depth == 0 || reversi.handleNoMoveAvailable(true))
            return new MoveResult(evaluateBoard(Main.cloneCharArray(reversi.board), this.playerMove, mode), -1, -1);

        int bestRow = -1;
        int bestCol = -1;
        int maxScore = Integer.MIN_VALUE;
        int minScore = Integer.MAX_VALUE;

        for (Move move : generateMoves(reversi.board, player)) {
            Reversi reversiCopy = new Reversi(Main.cloneCharArray(reversi.board), reversi.currentPlayer);

            reversiCopy.makeMove(move.row, move.col, player);
            MoveResult result = minimax(new Reversi(Main.cloneCharArray(reversiCopy.board), reversiCopy.currentPlayer), reversiCopy.currentPlayer, depth - 1, mode);
            nodesVisited++;
            nodesVisited += result.nodeVisited;
            int score = result.score;

            if (player == this.playerMove && score > maxScore) {
                maxScore = score;
                bestRow = move.row;
                bestCol = move.col;
            } else if (player != this.playerMove && score < minScore) {
                minScore = score;
                bestRow = move.row;
                bestCol = move.col;
            }

        }
        Instant end = Instant.now();
        return new MoveResult(player == this.playerMove ? maxScore : minScore, bestRow, bestCol, nodesVisited, Duration.between(start, end).toMillis());
    }

    public MoveResult minimaxalphabeta(Reversi reversi, char player, int depth, int alpha, int beta, String mode) {
        Instant start = Instant.now();
        int nodesVisited = 0;

        if (depth == 0 || reversi.handleNoMoveAvailable(true))
            return new MoveResult(evaluateBoard(Main.cloneCharArray(reversi.board), this.playerMove, mode), -1, -1);

        int maxScore = Integer.MIN_VALUE;
        int minScore = Integer.MAX_VALUE;
        int bestRow = -1;
        int bestCol = -1;

        for (Move move : generateMoves(reversi.board, player)) {
            Reversi reversiCopy = new Reversi(Main.cloneCharArray(reversi.board), reversi.currentPlayer);
            reversiCopy.makeMove(move.row, move.col, player);

            MoveResult result = minimaxalphabeta(new Reversi(Main.cloneCharArray(reversiCopy.board), reversiCopy.currentPlayer), reversiCopy.currentPlayer, depth - 1, alpha, beta, mode);
            nodesVisited++;
            nodesVisited += result.nodeVisited;
            int score = result.score;

            if (player == this.playerMove) {
                if (score > maxScore) {
                    maxScore = score;
                    bestRow = move.row;
                    bestCol = move.col;
                }
                alpha = Math.max(alpha, maxScore);
            } else {
                if (score < minScore) {
                    minScore = score;
                    bestRow = move.row;
                    bestCol = move.col;
                }
                beta = Math.min(beta, minScore);
            }
            if (beta <= alpha) {
                break;
            }

        }
        Instant end = Instant.now();
        return new MoveResult(player == this.playerMove ? maxScore : minScore, bestRow, bestCol, nodesVisited, Duration.between(start, end).toMillis());
    }

    public static ArrayList<Move> generateMoves(char[][] board, char playerColor) {
        ArrayList<Move> possibleMoves = new ArrayList<>();
        Reversi reversi = new Reversi(Main.cloneCharArray(board), playerColor);
        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 8; col++) {
                if (reversi.isValidMove(row, col, playerColor)) {
                    Move move = new Move(row, col);
                    possibleMoves.add(move);
                }
            }
        }
        return possibleMoves;
    }

    public int evaluateBoardDynamic(char[][] board, char player) {
        int score = 0;
        int[] table = new int[]{40, 20, 0};

        if (countEmptySpaces(board) > table[2])
            score += numberOfOwnedPieces(board, player);
        if (countEmptySpaces(board) < table[0])
            score += numberOfCapturedCornersAndEdges(board, player);
        if (countEmptySpaces(board) < table[1])
            score += numberOfPossibleMoves(board, player);
        return score;
    }

    public static int countEmptySpaces(char[][] board) {
        int emptySpaces = 0;
        for (char[] row : board) {
            for (char cell : row) {
                if (cell == ' ') {
                    emptySpaces++;
                }
            }
        }
        return emptySpaces;
    }

    public int evaluateBoard(char[][] board, char player, String mode) {

        if (mode.equals("pieces")) return numberOfOwnedPieces(board, player);
        if (mode.equals("moves")) return numberOfPossibleMoves(board, player);
        if (mode.equals("edges")) return numberOfCapturedCornersAndEdges(board, player);
        if (mode.equals("dynamic")) return evaluateBoardDynamic(board, player);
        return 0;
    }

    public int numberOfOwnedPieces(char[][] board, char player) {
        int numberOfPieces = 0;
        for (char[] chars : board) {
            for (char aChar : chars) {
                if (aChar == player) {
                    numberOfPieces++;
                }
            }
        }
        return numberOfPieces;
    }

    public int numberOfPossibleMoves(char[][] board, char player) {
        return generateMoves(board, player).size();
    }


    public int numberOfCapturedCornersAndEdges(char[][] board, char player) {
        int numberOfCorners = 0;
        numberOfCorners += countField(board, player, 0, 0, 1, 1);
        numberOfCorners += countField(board, player, 0, 7, 1, -1);
        numberOfCorners += countField(board, player, 7, 0, -1, 1);
        numberOfCorners += countField(board, player, 7, 7, -1, -1);

        return numberOfCorners;
    }

    public int countField(char[][] board, char player, int x, int y, int xPlus, int yPlus) {
        int numberOfCorners = 0;
        if (board[x][y] == player) {
            numberOfCorners += 10;
            for (int i = 1; i < 8; i++) {
                if (board[x][y + i * yPlus] == player) {
                    numberOfCorners += 6;
                } else break;
            }
            for (int i = 1; i < 8; i++) {
                if (board[x + i * xPlus][y] == player) {
                    numberOfCorners += 6;
                } else break;
            }
        }
        return numberOfCorners;
    }
}