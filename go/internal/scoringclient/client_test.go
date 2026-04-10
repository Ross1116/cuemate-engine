package scoringclient

import (
	"context"
	"errors"
	"net"
	"testing"
	"time"

	scoringv1 "github.com/Ross1116/cuemate-engine/go/gen/djengine/scoring/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type fakeScoringServer struct {
	scoringv1.UnimplementedScoringServiceServer
	metadataResp *scoringv1.GetScoringMetadataResponse
	scoreResp    *scoringv1.ScoreCandidateResponse
	scoreErr     error
}

func (f *fakeScoringServer) GetScoringMetadata(context.Context, *scoringv1.GetScoringMetadataRequest) (*scoringv1.GetScoringMetadataResponse, error) {
	return f.metadataResp, nil
}

func (f *fakeScoringServer) ScoreCandidate(context.Context, *scoringv1.ScoreCandidateRequest) (*scoringv1.ScoreCandidateResponse, error) {
	if f.scoreErr != nil {
		return nil, f.scoreErr
	}
	return f.scoreResp, nil
}

func (f *fakeScoringServer) GetRecommendations(context.Context, *scoringv1.GetRecommendationsRequest) (*scoringv1.GetRecommendationsResponse, error) {
	return &scoringv1.GetRecommendationsResponse{}, nil
}

func startFakeServer(t *testing.T, server scoringv1.ScoringServiceServer) string {
	t.Helper()

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}

	grpcServer := grpc.NewServer()
	scoringv1.RegisterScoringServiceServer(grpcServer, server)
	go func() {
		_ = grpcServer.Serve(lis)
	}()
	t.Cleanup(func() {
		grpcServer.Stop()
		_ = lis.Close()
	})

	return lis.Addr().String()
}

func TestLoadConfigUsesDefaults(t *testing.T) {
	t.Setenv("SCORING_GRPC_ADDR", "")
	t.Setenv("SCORING_RPC_TIMEOUT_MS", "")

	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}

	if cfg.Addr != DefaultScoringGRPCAddr {
		t.Fatalf("addr = %q, want %q", cfg.Addr, DefaultScoringGRPCAddr)
	}
	if cfg.RPCTimeout != DefaultScoringRPCTimeout {
		t.Fatalf("timeout = %v, want %v", cfg.RPCTimeout, DefaultScoringRPCTimeout)
	}
}

func TestLoadConfigFromEnv(t *testing.T) {
	t.Setenv("SCORING_GRPC_ADDR", "127.0.0.1:50099")
	t.Setenv("SCORING_RPC_TIMEOUT_MS", "1200")

	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}

	if cfg.Addr != "127.0.0.1:50099" {
		t.Fatalf("addr = %q", cfg.Addr)
	}
	if cfg.RPCTimeout != 1200*time.Millisecond {
		t.Fatalf("timeout = %v", cfg.RPCTimeout)
	}
}

func TestLoadConfigRejectsBadTimeout(t *testing.T) {
	t.Setenv("SCORING_RPC_TIMEOUT_MS", "oops")

	_, err := LoadConfig()
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestGetScoringMetadata(t *testing.T) {
	addr := startFakeServer(t, &fakeScoringServer{
		metadataResp: &scoringv1.GetScoringMetadataResponse{
			EngineVersion: "0.1.0",
			Components: []*scoringv1.ComponentMetadata{
				{ComponentId: "harmonic", Weight: 0.12, Active: true},
			},
		},
	})

	client, err := New(Config{Addr: addr, RPCTimeout: time.Second})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	defer client.Close()

	resp, err := client.GetScoringMetadata(context.Background(), &scoringv1.GetScoringMetadataRequest{})
	if err != nil {
		t.Fatalf("GetScoringMetadata() error = %v", err)
	}

	if got := resp.EngineVersion; got != "0.1.0" {
		t.Fatalf("EngineVersion = %q", got)
	}
	if len(resp.Components) != 1 || resp.Components[0].ComponentId != "harmonic" {
		t.Fatalf("Components = %#v", resp.Components)
	}
}

func TestScoreCandidateMapsFailedPrecondition(t *testing.T) {
	addr := startFakeServer(t, &fakeScoringServer{
		scoreErr: status.Error(codes.FailedPrecondition, "missing_signature_metadata"),
	})

	client, err := New(Config{Addr: addr, RPCTimeout: time.Second})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	defer client.Close()

	_, err = client.ScoreCandidate(context.Background(), &scoringv1.ScoreCandidateRequest{})
	if !errors.Is(err, ErrFailedPrecondition) {
		t.Fatalf("err = %v, want ErrFailedPrecondition", err)
	}
}

func TestScoreCandidateMapsInvalidArgument(t *testing.T) {
	addr := startFakeServer(t, &fakeScoringServer{
		scoreErr: status.Error(codes.InvalidArgument, "target_lane must be a supported lane."),
	})

	client, err := New(Config{Addr: addr, RPCTimeout: time.Second})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	defer client.Close()

	_, err = client.ScoreCandidate(context.Background(), &scoringv1.ScoreCandidateRequest{})
	if !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("err = %v, want ErrInvalidArgument", err)
	}
}

func TestNewFallsBackToDefaultAddr(t *testing.T) {
	var dialedAddr string

	client, err := New(
		Config{Addr: "", RPCTimeout: time.Millisecond},
		grpc.WithBlock(),
		grpc.WithContextDialer(func(ctx context.Context, addr string) (net.Conn, error) {
			dialedAddr = addr
			return nil, errors.New("expected dial stop")
		}),
	)
	if client != nil {
		defer client.Close()
	}
	if err == nil {
		t.Fatal("expected dial error")
	}
	if dialedAddr != DefaultScoringGRPCAddr {
		t.Fatalf("dialed addr = %q, want %q", dialedAddr, DefaultScoringGRPCAddr)
	}
}
