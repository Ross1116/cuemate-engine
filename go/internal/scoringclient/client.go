package scoringclient

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	scoringv1 "github.com/Ross1116/cuemate-engine/go/gen/djengine/scoring/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

const (
	DefaultScoringGRPCAddr   = "127.0.0.1:47834"
	DefaultScoringRPCTimeout = 250 * time.Millisecond
	defaultDialTimeout       = 2 * time.Second
)

var (
	ErrInvalidArgument    = errors.New("scoring request invalid")
	ErrFailedPrecondition = errors.New("scoring precondition failed")
)

type Client interface {
	GetScoringMetadata(ctx context.Context, req *scoringv1.GetScoringMetadataRequest, opts ...grpc.CallOption) (*scoringv1.GetScoringMetadataResponse, error)
	ScoreCandidate(ctx context.Context, req *scoringv1.ScoreCandidateRequest, opts ...grpc.CallOption) (*scoringv1.ScoreCandidateResponse, error)
	GetRecommendations(ctx context.Context, req *scoringv1.GetRecommendationsRequest, opts ...grpc.CallOption) (*scoringv1.GetRecommendationsResponse, error)
	GetFeedbackSummary(ctx context.Context, req *scoringv1.GetFeedbackSummaryRequest, opts ...grpc.CallOption) (*scoringv1.GetFeedbackSummaryResponse, error)
	Close() error
}

type Config struct {
	Addr       string
	RPCTimeout time.Duration
}

type grpcClient struct {
	conn       *grpc.ClientConn
	client     scoringv1.ScoringServiceClient
	rpcTimeout time.Duration
}

func LoadConfig() (Config, error) {
	addr := strings.TrimSpace(os.Getenv("SCORING_GRPC_ADDR"))
	if addr == "" {
		addr = DefaultScoringGRPCAddr
	}

	timeout := DefaultScoringRPCTimeout
	rawTimeout := strings.TrimSpace(os.Getenv("SCORING_RPC_TIMEOUT_MS"))
	if rawTimeout != "" {
		timeoutMS, err := strconv.Atoi(rawTimeout)
		if err != nil {
			return Config{}, fmt.Errorf("parse SCORING_RPC_TIMEOUT_MS: %w", err)
		}
		if timeoutMS <= 0 {
			return Config{}, fmt.Errorf("SCORING_RPC_TIMEOUT_MS must be positive, got %d", timeoutMS)
		}
		timeout = time.Duration(timeoutMS) * time.Millisecond
	}

	return Config{
		Addr:       addr,
		RPCTimeout: timeout,
	}, nil
}

func New(cfg Config, dialOptions ...grpc.DialOption) (Client, error) {
	if strings.TrimSpace(cfg.Addr) == "" {
		cfg.Addr = DefaultScoringGRPCAddr
	}
	if cfg.RPCTimeout <= 0 {
		cfg.RPCTimeout = DefaultScoringRPCTimeout
	}

	// This timeout matters when a caller opts into grpc.WithBlock().
	dialCtx, cancel := context.WithTimeout(context.Background(), defaultDialTimeout)
	defer cancel()

	options := []grpc.DialOption{grpc.WithTransportCredentials(insecure.NewCredentials())}
	options = append(options, dialOptions...)
	conn, err := grpc.DialContext(dialCtx, cfg.Addr, options...)
	if err != nil {
		return nil, fmt.Errorf("dial scoring service %s: %w", cfg.Addr, err)
	}

	return &grpcClient{
		conn:       conn,
		client:     scoringv1.NewScoringServiceClient(conn),
		rpcTimeout: cfg.RPCTimeout,
	}, nil
}

func (c *grpcClient) Close() error {
	return c.conn.Close()
}

func (c *grpcClient) GetScoringMetadata(ctx context.Context, req *scoringv1.GetScoringMetadataRequest, opts ...grpc.CallOption) (*scoringv1.GetScoringMetadataResponse, error) {
	ctx, cancel := c.withTimeout(ctx)
	defer cancel()
	resp, err := c.client.GetScoringMetadata(ctx, req, opts...)
	return resp, mapRPCError(err)
}

func (c *grpcClient) ScoreCandidate(ctx context.Context, req *scoringv1.ScoreCandidateRequest, opts ...grpc.CallOption) (*scoringv1.ScoreCandidateResponse, error) {
	ctx, cancel := c.withTimeout(ctx)
	defer cancel()
	resp, err := c.client.ScoreCandidate(ctx, req, opts...)
	return resp, mapRPCError(err)
}

func (c *grpcClient) GetRecommendations(ctx context.Context, req *scoringv1.GetRecommendationsRequest, opts ...grpc.CallOption) (*scoringv1.GetRecommendationsResponse, error) {
	ctx, cancel := c.withTimeout(ctx)
	defer cancel()
	resp, err := c.client.GetRecommendations(ctx, req, opts...)
	return resp, mapRPCError(err)
}

func (c *grpcClient) GetFeedbackSummary(ctx context.Context, req *scoringv1.GetFeedbackSummaryRequest, opts ...grpc.CallOption) (*scoringv1.GetFeedbackSummaryResponse, error) {
	ctx, cancel := c.withTimeout(ctx)
	defer cancel()
	resp, err := c.client.GetFeedbackSummary(ctx, req, opts...)
	return resp, mapRPCError(err)
}

func (c *grpcClient) withTimeout(ctx context.Context) (context.Context, context.CancelFunc) {
	if _, ok := ctx.Deadline(); ok {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, c.rpcTimeout)
}

func mapRPCError(err error) error {
	if err == nil {
		return nil
	}

	st, ok := status.FromError(err)
	if !ok {
		return err
	}

	switch st.Code() {
	case codes.InvalidArgument:
		return fmt.Errorf("%w: %s", ErrInvalidArgument, st.Message())
	case codes.FailedPrecondition:
		return fmt.Errorf("%w: %s", ErrFailedPrecondition, st.Message())
	default:
		return err
	}
}
