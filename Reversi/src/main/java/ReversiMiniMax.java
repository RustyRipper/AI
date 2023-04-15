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

        public MoveResult(int score, int row, int col) {
            this.score = score;
            this.row = row;
            this.col = col;
        }
    }

    public MoveResult minimax(Reversi reversi, char player, int depth) {

        if (depth == 0 || reversi.handleNoMoveAvailable(true))
            return new MoveResult(evaluateBoard(Main.cloneCharArray(reversi.board), this.playerMove), -1, -1);

        int bestRow = -1;
        int bestCol = -1;
        int maxScore = Integer.MIN_VALUE;
        int minScore = Integer.MAX_VALUE;
        for (Move move : generateMoves(reversi.board, player)) {
            Reversi reversiCopy = new Reversi(Main.cloneCharArray(reversi.board), reversi.currentPlayer);

            if (reversiCopy.isValidMove(move.row, move.col, player)) {
                reversiCopy.makeMove(move.row, move.col, player);
                MoveResult result = minimax(
                        new Reversi(Main.cloneCharArray(reversiCopy.board), reversiCopy.currentPlayer),
                        reversiCopy.currentPlayer,
                        depth - 1);
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
        }
        return new MoveResult(player == this.playerMove ? maxScore : minScore, bestRow, bestCol);
    }

    public MoveResult minimaxalphabeta(Reversi reversi, char player, int depth, int alpha, int beta) {

        if (depth == 0 || reversi.handleNoMoveAvailable(true))
            return new MoveResult(evaluateBoard(Main.cloneCharArray(reversi.board), this.playerMove), -1, -1);

        int maxScore = Integer.MIN_VALUE;
        int minScore = Integer.MAX_VALUE;
        int bestRow = -1;
        int bestCol = -1;

        for (Move move : generateMoves(reversi.board, player)) {
            Reversi reversiCopy = new Reversi(Main.cloneCharArray(reversi.board), reversi.currentPlayer);

            if (reversiCopy.isValidMove(move.row, move.col, player)) {
                reversiCopy.makeMove(move.row, move.col, player);
                MoveResult result = minimaxalphabeta(
                        new Reversi(Main.cloneCharArray(reversiCopy.board), reversiCopy.currentPlayer),
                        reversiCopy.currentPlayer,
                        depth - 1,
                        alpha,
                        beta);
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
        }
        return new MoveResult(player == this.playerMove ? maxScore : minScore, bestRow, bestCol);
    }

    private ArrayList<Move> generateMoves(char[][] board, char playerColor) {
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

    public int evaluateBoard(char[][] board, char player) {
        int score = 0;

        score += numberOfOwnedPieces(board, player);

        score += numberOfPossibleMoves(board, player);

        score += numberOfCapturedCorners(board, player);

        return score;
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


    public int numberOfCapturedCorners(char[][] board, char player) {
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
            numberOfCorners++;
            if (board[x][y + yPlus] == player)
                numberOfCorners += 1;

            if (board[x + xPlus][y] == player)
                numberOfCorners += 1;
        }
        return numberOfCorners;
    }
}