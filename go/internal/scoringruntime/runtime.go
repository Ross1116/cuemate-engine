package scoringruntime

import (
	"context"
	"errors"
	"fmt"
	"sync"

	scoringv1 "github.com/Ross1116/cuemate-engine/go/gen/djengine/scoring/v1"
	"github.com/Ross1116/cuemate-engine/go/internal/scoringclient"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

var (
	ErrCircuitOpen = errors.New("scoring circuit open")
)

type Client interface {
	GetScoringMetadata(ctx context.Context, req *scoringv1.GetScoringMetadataRequest, opts ...grpc.CallOption) (*scoringv1.GetScoringMetadataResponse, error)
	GetRecommendations(ctx context.Context, req *scoringv1.GetRecommendationsRequest, opts ...grpc.CallOption) (*scoringv1.GetRecommendationsResponse, error)
	GetFeedbackSummary(ctx context.Context, req *scoringv1.GetFeedbackSummaryRequest, opts ...grpc.CallOption) (*scoringv1.GetFeedbackSummaryResponse, error)
	Close() error
}

type Runtime struct {
	client        Client
	failThreshold int

	mu                  sync.Mutex
	cachedMetadata      *scoringv1.GetScoringMetadataResponse
	consecutiveFailures int
	breakerOpen         bool
}

func New(client Client, failThreshold int) *Runtime {
	if failThreshold <= 0 {
		failThreshold = 3
	}
	return &Runtime{client: client, failThreshold: failThreshold}
}

func NewDefault() (*Runtime, error) {
	cfg, err := scoringclient.LoadConfig()
	if err != nil {
		return nil, err
	}
	client, err := scoringclient.New(cfg)
	if err != nil {
		return nil, err
	}
	return New(client, 3), nil
}

func (r *Runtime) Close() error {
	return r.client.Close()
}

func (r *Runtime) State() (bool, int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.breakerOpen, r.consecutiveFailures
}

func (r *Runtime) CachedMetadata() *scoringv1.GetScoringMetadataResponse {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.cachedMetadata
}

func (r *Runtime) RefreshMetadata(ctx context.Context) (*scoringv1.GetScoringMetadataResponse, error) {
	resp, err := r.client.GetScoringMetadata(ctx, &scoringv1.GetScoringMetadataRequest{})
	if err != nil {
		r.recordFailure(err)
		return nil, err
	}
	r.recordSuccess(resp)
	return resp, nil
}

func (r *Runtime) GetRecommendations(ctx context.Context, req *scoringv1.GetRecommendationsRequest) (*scoringv1.GetRecommendationsResponse, error) {
	if open, _ := r.State(); open {
		return nil, ErrCircuitOpen
	}
	rpcCtx, cancel := context.WithTimeout(ctx, scoringclient.DefaultScoringRPCTimeout)
	defer cancel()
	resp, err := r.client.GetRecommendations(rpcCtx, req)
	if err != nil {
		if !IsUnimplemented(err) {
			r.recordFailure(err)
		}
		return nil, err
	}
	r.resetFailures()
	return resp, nil
}

func (r *Runtime) GetFeedbackSummary(ctx context.Context, req *scoringv1.GetFeedbackSummaryRequest) (*scoringv1.GetFeedbackSummaryResponse, error) {
	if open, _ := r.State(); open {
		return nil, ErrCircuitOpen
	}
	rpcCtx, cancel := context.WithTimeout(ctx, scoringclient.DefaultScoringRPCTimeout)
	defer cancel()
	resp, err := r.client.GetFeedbackSummary(rpcCtx, req)
	if err != nil {
		if !IsUnimplemented(err) {
			r.recordFailure(err)
		}
		return nil, err
	}
	r.resetFailures()
	return resp, nil
}

func (r *Runtime) recordSuccess(resp *scoringv1.GetScoringMetadataResponse) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.cachedMetadata = resp
	r.consecutiveFailures = 0
	r.breakerOpen = false
}

func (r *Runtime) resetFailures() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.consecutiveFailures = 0
	r.breakerOpen = false
}

func (r *Runtime) recordFailure(err error) {
	if errors.Is(err, scoringclient.ErrInvalidArgument) || errors.Is(err, scoringclient.ErrFailedPrecondition) {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.consecutiveFailures++
	if r.consecutiveFailures >= r.failThreshold {
		r.breakerOpen = true
	}
}

func IsCompatibilityError(err error) bool {
	return errors.Is(err, scoringclient.ErrFailedPrecondition)
}

func IsRequestError(err error) bool {
	return errors.Is(err, scoringclient.ErrInvalidArgument)
}

func IsUnavailable(err error) bool {
	if errors.Is(err, ErrCircuitOpen) {
		return true
	}
	st, ok := status.FromError(err)
	if !ok || st == nil {
		return false
	}
	// Treat only transient transport/load conditions as temporarily unavailable.
	switch st.Code() {
	case codes.Unavailable, codes.DeadlineExceeded, codes.ResourceExhausted:
		return true
	default:
		return false
	}
}

func IsUnimplemented(err error) bool {
	st, ok := status.FromError(err)
	if !ok || st == nil {
		return false
	}
	return st.Code() == codes.Unimplemented
}

func DescribeUnavailable(err error) string {
	if errors.Is(err, ErrCircuitOpen) {
		return "Scoring service is temporarily unavailable; retry after the service recovers."
	}
	if err == nil {
		return ""
	}
	return fmt.Sprintf("Scoring service is temporarily unavailable: %v", err)
}
